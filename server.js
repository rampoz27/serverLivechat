// server.js
import express from "express";
import { createClient } from "@supabase/supabase-js";
import { OpenAI } from "openai";

const app = express();
app.use(express.json());
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*"); // longgarkan dulu untuk testing, ketatkan pas produksi
  res.header("Access-Control-Allow-Headers", "Content-Type");
  next();
});

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const openai = new OpenAI();

async function getEmbedding(text) {
  const res = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return res.data[0].embedding;
}

// ============================================
// ENDPOINT UTAMA: kembalikan BEBERAPA opsi balasan
// ============================================
app.post("/api/suggest-replies", async (req, res) => {
  const { customerMessage } = req.body;
  if (!customerMessage) {
    return res.status(400).json({ error: "Pesan kosong" });
  }

  try {
    // 1. Cari beberapa jawaban paling relevan langsung dari knowledge base
    const embedding = await getEmbedding(customerMessage);
    const { data: matches, error } = await supabase.rpc("match_knowledge", {
      query_embedding: embedding,
      match_count: 4, // ambil 4 kandidat jawaban
    });
    if (error) throw error;

    // Filter yang similarity-nya terlalu rendah (kemungkinan tidak relevan)
    const relevant = (matches || []).filter((m) => m.similarity > 0.75);

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

    // 2. Opsi 1: jawaban langsung dari knowledge base yang paling mirip (apa adanya)
    const directSuggestions = relevant.map((m) => ({
      answer: m.answer,
      source: "knowledge_base",
      similarity: m.similarity,
    }));

    // 3. Opsi tambahan: satu jawaban yang di-rewrite LLM supaya lebih natural,
    //    menggabungkan konteks dari beberapa match sekaligus
    const contextText = relevant
      .map((m) => `Q: ${m.question}\nA: ${m.answer}`)
      .join("\n\n");

    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `Kamu asisten customer service. Berdasarkan konteks berikut, tulis SATU jawaban singkat, ramah, dan natural untuk pelanggan. Jangan tambahkan info di luar konteks.\n\nKonteks:\n${contextText}`,
        },
        { role: "user", content: customerMessage },
      ],
    });

    const rewritten = {
      answer: completion.choices[0].message.content,
      source: "ai_rewrite",
    };

    // Taruh hasil rewrite AI di paling atas, disusul jawaban asli dari KB
    res.json({
      suggestions: [rewritten, ...directSuggestions],
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Gagal generate saran jawaban" });
  }
});

// ============================================
// ENDPOINT ADMIN: generate embedding untuk data yang belum punya
// Dilindungi ADMIN_SECRET supaya tidak sembarang orang bisa memicu ini
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
        const embedding = await getEmbedding(combinedText);
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
// HEALTH CHECK — buat cek server hidup atau tidak
// ============================================
app.get("/", (req, res) => {
  res.json({ status: "ok", message: "LiveChat RAG server jalan" });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`RAG server jalan di port ${PORT}`));
