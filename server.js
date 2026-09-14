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
// STOPWORDS — kata umum bahasa Indonesia yang diabaikan saat cari keyword
// ============================================
const STOPWORDS = new Set([
  "yang", "dan", "di", "ke", "dari", "ini", "itu", "saya", "kak", "kakak",
  "gimana", "bagaimana", "cara", "apa", "apakah", "untuk", "dengan", "ada",
  "bisa", "mau", "nya", "ya", "min", "kok", "gak", "ga", "tidak", "sih",
  "aja", "juga", "atau", "kalau", "kalo", "saja", "pada", "adalah", "akan"
]);

function extractKeywords(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .map((w) => w.trim())
    .filter((w) => w.length > 2 && !STOPWORDS.has(w));
}

// ============================================
// FALLBACK: cari lewat KATA KUNCI (kolom question + kategori)
// Dipakai kalau embedding gagal, atau similarity semantik tidak ketemu apa-apa
// ============================================
async function keywordFallbackSearch(customerMessage, limit = 4) {
  const words = extractKeywords(customerMessage);
  if (words.length === 0) return [];

  try {
    // 1. Cari lewat kategori/isi jawaban yang match salah satu kata kunci
    const answerFilter = words
      .flatMap((w) => [`kategori.ilike.%${w}%`, `answer.ilike.%${w}%`])
      .join(",");
    const { data: answerMatches } = await supabase
      .from("faq_answers")
      .select("id, answer, kategori")
      .or(answerFilter)
      .limit(limit * 2);

    // 2. Cari lewat isi pertanyaan (question) yang match salah satu kata kunci
    const questionFilter = words.map((w) => `question.ilike.%${w}%`).join(",");
    const { data: questionMatches } = await supabase
      .from("faq_questions")
      .select("faq_answer_id, question")
      .or(questionFilter)
      .limit(limit * 2);

    let extraAnswers = [];
    const answerIdsFromQuestions = [...new Set((questionMatches || []).map((q) => q.faq_answer_id))];
    if (answerIdsFromQuestions.length > 0) {
      const { data } = await supabase
        .from("faq_answers")
        .select("id, answer, kategori")
        .in("id", answerIdsFromQuestions);
      extraAnswers = data || [];
    }

    // Gabung & hilangkan duplikat berdasarkan id jawaban
    const combinedMap = new Map();
    [...(answerMatches || []), ...extraAnswers].forEach((a) => combinedMap.set(a.id, a));

    return Array.from(combinedMap.values())
      .slice(0, limit)
      .map((a) => ({
        answer: a.answer,
        source: "keyword_fallback",
        kategori: a.kategori,
      }));
  } catch (err) {
    console.error("Keyword fallback gagal:", err);
    return [];
  }
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
      // Semantik tidak nemu apa-apa — coba fallback kata kunci sebelum nyerah total
      const keywordResults = await keywordFallbackSearch(customerMessage);
      if (keywordResults.length > 0) {
        return res.json({ suggestions: keywordResults });
      }
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
    console.error("Embedding/RPC gagal, coba fallback kata kunci:", err);

    // Embedding atau RPC gagal total (misal model error, Supabase RPC bermasalah) —
    // tetap coba kasih saran lewat pencarian kata kunci biasa
    try {
      const keywordResults = await keywordFallbackSearch(customerMessage);
      if (keywordResults.length > 0) {
        return res.json({ suggestions: keywordResults });
      }
    } catch (fallbackErr) {
      console.error("Fallback kata kunci juga gagal:", fallbackErr);
    }

    res.status(500).json({ error: "Gagal generate saran jawaban" });
  }
});

// ============================================
// ENDPOINT: daftar semua kategori unik yang ada
// ============================================
app.get("/api/categories", async (req, res) => {
  try {
    const { data, error } = await supabase
      .from("faq_answers")
      .select("kategori")
      .not("kategori", "is", null);

    if (error) throw error;

    const uniqueCategories = [...new Set((data || []).map((d) => d.kategori))].filter(Boolean).sort();
    res.json({ categories: uniqueCategories });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Gagal ambil daftar kategori" });
  }
});

// ============================================
// ENDPOINT: cari jawaban berdasarkan kategori (manual, tanpa embedding)
// Cara pakai: GET /api/faqs-by-category?kategori=pengiriman
// ============================================
app.get("/api/faqs-by-category", async (req, res) => {
  const { kategori } = req.query;
  if (!kategori) {
    return res.status(400).json({ error: "Parameter kategori kosong" });
  }

  try {
    const { data: answers, error } = await supabase
      .from("faq_answers")
      .select("id, answer, kategori")
      .eq("kategori", kategori);

    if (error) throw error;

    if (!answers || answers.length === 0) {
      return res.json({ suggestions: [] });
    }

    // Ambil salah satu contoh pertanyaan per jawaban, biar agent tahu konteksnya
    const answerIds = answers.map((a) => a.id);
    const { data: questions } = await supabase
      .from("faq_questions")
      .select("faq_answer_id, question")
      .in("faq_answer_id", answerIds);

    const questionMap = {};
    (questions || []).forEach((q) => {
      if (!questionMap[q.faq_answer_id]) questionMap[q.faq_answer_id] = q.question;
    });

    const suggestions = answers.map((a) => ({
      answer: a.answer,
      source: "kategori",
      kategori: a.kategori,
      contoh_pertanyaan: questionMap[a.id] || null,
    }));

    res.json({ suggestions });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Gagal ambil jawaban berdasarkan kategori" });
  }
});

// ============================================
// ENDPOINT ADMIN: generate embedding LOKAL untuk pertanyaan yang belum punya
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
      .from("faq_questions")
      .select("id, question, faq_answer_id")
      .is("embedding", null);

    if (error) throw error;

    if (!rows || rows.length === 0) {
      return res.json({ message: "Tidak ada pertanyaan yang perlu di-generate embedding-nya.", processed: 0 });
    }

    // Ambil semua jawaban terkait dalam 1 query terpisah (lebih aman daripada join)
    const answerIds = [...new Set(rows.map((r) => r.faq_answer_id))];
    const { data: answerRows, error: answerFetchError } = await supabase
      .from("faq_answers")
      .select("id, answer")
      .in("id", answerIds);

    if (answerFetchError) throw answerFetchError;

    const answerMap = {};
    (answerRows || []).forEach((a) => {
      answerMap[a.id] = a.answer;
    });

    const results = [];
    for (const row of rows) {
      const answerText = answerMap[row.faq_answer_id] || "";
      const combinedText = `${row.question}\n${answerText}`;
      try {
        const embedding = await getEmbeddingLocal(combinedText);
        const { error: updateError } = await supabase
          .from("faq_questions")
          .update({ embedding })
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
// ENDPOINT ADMIN: bulk-insert FAQ (1 jawaban + banyak variasi pertanyaan)
// Insert ke 2 tabel: faq_answers (sekali) lalu faq_questions (banyak baris)
// Dipanggil dari halaman form /admin/bulk-insert (lihat di bawah)
// ============================================
app.post("/admin/bulk-insert-faq", async (req, res) => {
  const { secret, groups } = req.body;
  if (!process.env.ADMIN_SECRET || secret !== process.env.ADMIN_SECRET) {
    return res.status(403).json({ error: "Secret salah atau belum di-set" });
  }
  if (!Array.isArray(groups) || groups.length === 0) {
    return res.status(400).json({ error: "Format 'groups' tidak valid atau kosong" });
  }

  const results = [];

  for (const group of groups) {
    const { answer, kategori, question_variations } = group;
    if (!answer || !Array.isArray(question_variations) || question_variations.length === 0) {
      results.push({ status: "dilewati", reason: "answer atau question_variations kosong", group });
      continue;
    }

    // 1. Insert jawabannya SEKALI ke faq_answers
    const { data: answerRow, error: answerError } = await supabase
      .from("faq_answers")
      .insert({ answer, kategori: kategori || null })
      .select("id")
      .single();

    if (answerError) {
      results.push({ status: "gagal", reason: "gagal insert jawaban: " + answerError.message, group });
      continue;
    }

    // 2. Insert tiap variasi pertanyaan ke faq_questions, link ke jawaban di atas
    for (const question of question_variations) {
      try {
        const embedding = await getEmbeddingLocal(`${question}\n${answer}`);
        const { error } = await supabase
          .from("faq_questions")
          .insert({ faq_answer_id: answerRow.id, question, embedding });

        if (error) throw error;
        results.push({ status: "ok", question });
      } catch (err) {
        results.push({ status: "gagal", question, error: err.message });
      }
    }
  }

  const okCount = results.filter((r) => r.status === "ok").length;
  res.json({ message: `Selesai. ${okCount} baris pertanyaan berhasil ditambahkan.`, results });
});

// ============================================
// HALAMAN FORM ADMIN: input FAQ langsung dari browser, tanpa perlu coding
// Buka: https://server-kamu.onrender.com/admin/bulk-insert
// ============================================
app.get("/admin/bulk-insert", (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bulk Insert FAQ</title>
<style>
  body { font-family: sans-serif; max-width: 700px; margin: 20px auto; padding: 0 16px; background: #111; color: #eee; }
  h1 { font-size: 18px; }
  label { display: block; margin-top: 14px; margin-bottom: 4px; font-size: 13px; color: #aaa; }
  input, textarea { width: 100%; box-sizing: border-box; padding: 10px; border-radius: 6px; border: 1px solid #444; background: #1c1c1c; color: #eee; font-family: monospace; font-size: 13px; }
  textarea { min-height: 260px; }
  button { margin-top: 16px; padding: 12px 20px; background: #1a73e8; color: white; border: none; border-radius: 6px; font-size: 14px; cursor: pointer; }
  button:disabled { background: #555; }
  #result { margin-top: 16px; padding: 12px; background: #1c1c1c; border-radius: 6px; white-space: pre-wrap; font-size: 12px; max-height: 300px; overflow-y: auto; }
  .hint { font-size: 12px; color: #888; margin-top: 4px; }
</style>
</head>
<body>
  <h1>📥 Bulk Insert FAQ ke Knowledge Base</h1>

  <label>Admin Secret</label>
  <input type="password" id="secret" placeholder="Isi ADMIN_SECRET kamu">

  <label>Data FAQ (format JSON)</label>
  <div class="hint">Boleh isi lebih dari 1 grup. Tiap grup = 1 jawaban + banyak variasi pertanyaan.</div>
  <textarea id="groups">[
  {
    "answer": "Refund bisa diajukan maksimal 7 hari setelah pembelian melalui menu Akun > Riwayat Pesanan.",
    "kategori": "kebijakan",
    "question_variations": [
      "Bagaimana cara refund?",
      "Gimana cara ngembaliin barang?",
      "Mau refund gimana caranya kak?",
      "Barang mau dikembalikan, prosesnya gimana?"
    ]
  }
]</textarea>

  <button id="submitBtn" onclick="submitData()">Simpan ke Knowledge Base</button>

  <div id="result"></div>

<script>
async function submitData() {
  const btn = document.getElementById('submitBtn');
  const resultEl = document.getElementById('result');
  const secret = document.getElementById('secret').value;
  let groups;

  try {
    groups = JSON.parse(document.getElementById('groups').value);
  } catch (e) {
    resultEl.textContent = 'JSON tidak valid: ' + e.message;
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Memproses... (mungkin beberapa detik)';
  resultEl.textContent = '';

  try {
    const res = await fetch('/admin/bulk-insert-faq', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ secret, groups })
    });
    const data = await res.json();
    resultEl.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    resultEl.textContent = 'Gagal: ' + e.message;
  }

  btn.disabled = false;
  btn.textContent = 'Simpan ke Knowledge Base';
}
</script>
</body>
</html>`);
});

// ============================================
// HEALTH CHECK
// ============================================
app.get("/", (req, res) => {
  res.json({ status: "ok", message: "LiveChat RAG server jalan (embedding lokal + Gemini opsional)" });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`RAG server jalan di port ${PORT}`));
