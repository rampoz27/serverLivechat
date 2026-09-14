// server.js
import express from "express";
import { createClient } from "@supabase/supabase-js";
import { GoogleGenAI } from "@google/genai";
import { pipeline, env } from "@huggingface/transformers";

const app = express();
app.use(express.json({ limit: "2mb" })); // dinaikkan sedikit karena body bisa berisi vektor 384 angka
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*"); // longgarkan dulu untuk testing, ketatkan pas produksi
  res.header("Access-Control-Allow-Headers", "Content-Type");
  next();
});

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

// Gemini cuma dipakai untuk REWRITE jawaban jadi lebih natural (opsional, jarang dipanggil)
const ai = process.env.GEMINI_API_KEY ? new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY }) : null;
const CHAT_MODEL = "gemini-flash-latest";

// ============================================
// MODEL EMBEDDING LOKAL (jalan di server, sama persis dengan yang dipakai di browser HP)
// Model kecil (~23MB), di-load sekali lalu di-cache di memori proses
// ============================================
const EMBEDDING_MODEL_ID = "Xenova/all-MiniLM-L6-v2";
let embedderPromise = null;

function getEmbedder() {
  if (!embedderPromise) {
    env.allowLocalModels = false; // ambil dari Hugging Face Hub, bukan file lokal
    embedderPromise = pipeline("feature-extraction", EMBEDDING_MODEL_ID, { dtype: "q8" });
  }
  return embedderPromise;
}

async function getEmbeddingLocal(text) {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

async function generateReply(systemPrompt, userMessage) {
  if (!ai) return null; // Gemini belum dikonfigurasi, skip rewrite
  const response = await ai.models.generateContent({
    model: CHAT_MODEL,
    contents: `${systemPrompt}\n\nPertanyaan pelanggan: ${userMessage}`,
  });
  return response.text;
}

// ============================================
// ENDPOINT UTAMA: terima VEKTOR (dihitung di HP) + teks asli untuk konteks rewrite
// ============================================
app.post("/api/suggest-replies", async (req, res) => {
  const { customerMessage, embedding } = req.body;

  if (!customerMessage) {
    return res.status(400).json({ error: "Pesan kosong" });
  }

  try {
    // Kalau browser TIDAK kirim embedding (misal model lokal gagal load),
    // fallback ke hitung embedding di server sebagai cadangan
    const queryEmbedding =
      Array.isArray(embedding) && embedding.length === 384
        ? embedding
        : await getEmbeddingLocal(customerMessage);

    // 1. Cari beberapa jawaban paling relevan dari knowledge base
    const { data: matches, error } = await supabase.rpc("match_knowledge", {
      query_embedding: queryEmbedding,
      match_count: 4,
    });
    if (error) throw error;

    const relevant = (matches || []).filter((m) => m.similarity > 0.6);

    if (relevant.length === 0) {
      return res.json({
        suggestions: [
          {
            answer:
              "Maaf, saya belum menemukan jawaban pasti untuk pertanyaan ini. Mohon tunggu, akan saya cek lebih lanjut ya.",
            source: "fallback",
          },
        ],
      });
    }

    const directSuggestions = relevant.map((m) => ({
      answer: m.answer,
      source: "knowledge_base",
      similarity: m.similarity,
    }));

    // 2. Kalau similarity TERTINGGI sudah sangat bagus (>0.88), SKIP panggilan Gemini
    //    sama sekali — langsung pakai jawaban asli. Ini menghemat kuota Gemini besar-besaran.
    const bestScore = relevant[0].similarity;
    let rewritten = null;

    if (bestScore < 0.88 && ai) {
      const contextText = relevant
        .map((m) => `Q: ${m.question}\nA: ${m.answer}`)
        .join("\n\n");
      const systemPrompt = `Kamu asisten customer service. Berdasarkan konteks berikut, tulis SATU jawaban singkat, ramah, dan natural untuk pelanggan. Jangan tambahkan info di luar konteks.\n\nKonteks:\n${contextText}`;
      const rewrittenText = await generateReply(systemPrompt, customerMessage);
      if (rewrittenText) {
        rewritten = { answer: rewrittenText, source: "ai_rewrite" };
      }
    }

    const suggestions = rewritten ? [rewritten, ...directSuggestions] : directSuggestions;
    res.json({ suggestions });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Gagal generate saran jawaban" });
  }
});

// ============================================
// ENDPOINT ADMIN: generate embedding LOKAL untuk data KB yang belum punya
// Cara pakai: buka di browser
//   https://server-kamu.onrender.com/admin/generate-embeddings?secret=ISI_SECRET_KAMU
// ============================================
app.get("/admin/generate-embeddings", async (req, res) => {
  const { secret } = req.query;
  if (!process.env.ADMIN_SECRET || secret !== process.env.ADMIN_SECRET) {
    return res.status(403).json({ error: "Secret salah atau belum di-set" });
  }

  try {
    const { data: rows, error } = await supabase
      .from("knowledge_base")
      .select("id, question, answer")
      .is("embedding", null);

    if (error) throw error;

    if (!rows || rows.length === 0) {
      return res.json({ message: "Tidak ada data yang perlu di-generate embedding-nya.", processed: 0 });
    }

    const results = [];
    for (const row of rows) {
      const combinedText = `${row.question}\n${row.answer}`;
      try {
        const embedding = await getEmbeddingLocal(combinedText);
        const { error: updateError } = await supabase
          .from("knowledge_base")
          .update({ embedding, updated_at: new Date().toISOString() })
          .eq("id", row.id);

        if (updateError) throw updateError;
        results.push({ id: row.id, status: "ok" });
      } catch (err) {
        results.push({ id: row.id, status: "gagal", error: err.message });
      }
    }

    res.json({ message: "Selesai memproses.", processed: results.length, results });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Gagal menjalankan proses generate embedding" });
  }
});

// ============================================
// HEALTH CHECK
// ============================================
app.get("/", (req, res) => {
  res.json({ status: "ok", message: "LiveChat RAG server jalan (embedding lokal + Gemini opsional)" });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`RAG server jalan di port ${PORT}`));
