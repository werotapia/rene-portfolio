// /api/match — Vercel serverless function (Node.js runtime).
//
// Takes a job description pasted by a recruiter, sends it together with
// Rene's profile.json to Groq's free API, and returns a structured
// "job fit" report as JSON. The Groq API key lives only in the
// GROQ_API_KEY environment variable on Vercel — it is never sent to the
// browser.

const fs = require('fs');
const path = require('path');

const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
// Best free-tier text model on Groq as of Sept 2026 (checked against Groq's
// own docs — Llama 3.3 70B was removed from the free tier in Aug 2026).
const GROQ_MODEL = 'openai/gpt-oss-120b';

const MAX_JOB_DESCRIPTION_CHARS = 8000;
const MIN_JOB_DESCRIPTION_CHARS = 30;
const MAX_REPORTS_PER_VISITOR_PER_HOUR = 5;
const HOUR_MS = 60 * 60 * 1000;

const BUSY_MESSAGE =
  'El asistente está ocupado, intenta en unos minutos o contáctame directamente. ' +
  '/ The assistant is busy right now, please try again in a few minutes or contact me directly.';

const RATE_LIMIT_MESSAGE =
  'Ya generaste 5 reportes en la última hora. Intenta de nuevo más tarde o contáctame directamente. ' +
  '/ You’ve already generated 5 reports in the last hour. Please try again later, or contact me directly.';

const TOO_LONG_MESSAGE =
  'La descripción del puesto es demasiado larga (máximo 8,000 caracteres). Por favor acórtala e intenta de nuevo. ' +
  '/ The job description is too long (8,000 character limit). Please shorten it and try again.';

const TOO_SHORT_MESSAGE =
  'Pega una descripción de puesto más completa para poder analizarla. ' +
  '/ Please paste a fuller job description so it can be analyzed.';

// ---- best-effort in-memory rate limiter -----------------------------------
// This resets whenever Vercel spins up a fresh instance of the function
// (cold start), so it is a soft, best-effort layer, not a hard guarantee.
// It's paired with a client-side localStorage check for the everyday case,
// and Groq's own account-wide free-tier limit as the real backstop.
const visitorHits = new Map();

function isRateLimited(ip) {
  const now = Date.now();
  const hits = (visitorHits.get(ip) || []).filter((t) => now - t < HOUR_MS);
  if (hits.length >= MAX_REPORTS_PER_VISITOR_PER_HOUR) {
    visitorHits.set(ip, hits);
    return true;
  }
  hits.push(now);
  visitorHits.set(ip, hits);
  return false;
}

// Occasionally trim the map so it doesn't grow forever on a long-lived warm
// instance.
function pruneVisitorHits() {
  const now = Date.now();
  for (const [ip, hits] of visitorHits) {
    const fresh = hits.filter((t) => now - t < HOUR_MS);
    if (fresh.length === 0) visitorHits.delete(ip);
    else visitorHits.set(ip, fresh);
  }
}

function getClientIp(req) {
  const fwd = req.headers['x-forwarded-for'];
  if (typeof fwd === 'string' && fwd.length) return fwd.split(',')[0].trim();
  return req.socket && req.socket.remoteAddress ? req.socket.remoteAddress : 'unknown';
}

// ---- profile loading + a compact digest for the prompt --------------------
let cachedProfile = null;
function loadProfile() {
  if (cachedProfile) return cachedProfile;
  const raw = fs.readFileSync(path.join(process.cwd(), 'profile.json'), 'utf8');
  cachedProfile = JSON.parse(raw);
  return cachedProfile;
}

// Send a condensed version of profile.json to the model rather than the
// full file — it keeps every fact, just trims the longest narrative prose,
// which matters because Groq's free tier caps tokens per minute.
function buildProfileDigest(p) {
  return {
    name: p.name,
    headline: p.headline,
    location: p.location,
    seeking: p.seeking,
    summary: p.summary,
    work_authorization: p.work_authorization,
    availability: p.availability,
    relocation: p.relocation,
    countries: p.countries,
    languages: p.languages,
    education: (p.education || []).map((e) => ({
      school: e.school,
      credential: e.credential,
      location: e.location,
      dates: e.dates,
      details: e.details,
    })),
    experience: (p.experience || []).map((e) => ({
      organization: e.organization,
      roles: e.roles,
      location: e.location,
      dates: e.dates,
      status: e.status,
      status_note: e.status_note,
      bullets: e.bullets,
      project_names_and_results: (e.projects || []).map((pr) => `${pr.name}: ${pr.result}`),
      tags: e.tags,
    })),
    skills: p.skills,
    industries: (p.industries || []).map((i) => `${i.name} — ${i.note}`),
    certifications: p.certifications,
    key_achievements: p.key_achievements,
  };
}

const SYSTEM_PROMPT = `You are a hiring-fit assistant embedded on Rene Tapia Rivera's personal portfolio site.
You will receive Rene's profile data as JSON, then a job description pasted by a recruiter or hiring manager.

Assess how well Rene's background fits THIS specific job, using ONLY the profile data given to you.

Strict rules:
- Never invent, assume, or exaggerate any employer, title, skill, certification, metric, or result that is not explicitly present in the profile data.
- Any field marked "TODO" in the profile is information Rene has not provided yet — treat it as unknown, never guess a value for it.
- Any experience entry with "status": "upcoming" has NOT happened yet — never describe it as completed experience; you may only mention it as planned/upcoming work.
- Be honest and specific about genuine gaps between the job and the profile. Do not oversell. match_percentage should reflect real fit and must NOT default to a high number.
- Detect whether the job description is written in Spanish or English, and write every string in your JSON response in that same language.
- Output ONLY one valid JSON object. No markdown code fences, no commentary before or after it.

Respond with exactly this schema:
{
  "language": "en" or "es",
  "match_percentage": integer from 0 to 100,
  "summary": "2-3 sentence summary of overall fit, in the detected language",
  "matched_requirements": [ { "requirement": "a requirement from the job posting", "evidence": "specific evidence from Rene's profile" } ],
  "transferable_skills": [ { "skill": "skill name", "note": "why it's relevant to this job" } ],
  "growth_areas": [ { "area": "area name", "note": "honest, constructive note" } ]
}
Include 3-6 items in matched_requirements, 2-5 in transferable_skills, and 1-4 in growth_areas.`;

// ---- JSON extraction + schema validation -----------------------------------
function extractJson(text) {
  try {
    return JSON.parse(text);
  } catch (_) {
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}');
    if (start === -1 || end === -1 || end <= start) return null;
    try {
      return JSON.parse(text.slice(start, end + 1));
    } catch (_e) {
      return null;
    }
  }
}

function isNonEmptyString(v) {
  return typeof v === 'string' && v.trim().length > 0;
}

function validateReport(r) {
  if (!r || typeof r !== 'object') return false;
  if (r.language !== 'en' && r.language !== 'es') return false;
  if (typeof r.match_percentage !== 'number' || r.match_percentage < 0 || r.match_percentage > 100) return false;
  if (!isNonEmptyString(r.summary)) return false;
  if (!Array.isArray(r.matched_requirements) || r.matched_requirements.length === 0) return false;
  if (!r.matched_requirements.every((m) => m && isNonEmptyString(m.requirement) && isNonEmptyString(m.evidence))) return false;
  if (!Array.isArray(r.transferable_skills)) return false;
  if (!r.transferable_skills.every((s) => s && isNonEmptyString(s.skill) && isNonEmptyString(s.note))) return false;
  if (!Array.isArray(r.growth_areas)) return false;
  if (!r.growth_areas.every((g) => g && isNonEmptyString(g.area) && isNonEmptyString(g.note))) return false;
  return true;
}

// ---- handler ----------------------------------------------------------------
module.exports = async function handler(req, res) {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');

  if (req.method !== 'POST') {
    res.status(405).json({ ok: false, error: 'method_not_allowed', message: 'Use POST.' });
    return;
  }

  let body = req.body;
  if (typeof body === 'string') {
    try {
      body = JSON.parse(body);
    } catch (_) {
      body = {};
    }
  }
  if (!body || typeof body !== 'object') body = {};

  const jobDescription = typeof body.jobDescription === 'string' ? body.jobDescription.trim() : '';

  if (jobDescription.length < MIN_JOB_DESCRIPTION_CHARS) {
    res.status(400).json({ ok: false, error: 'too_short', message: TOO_SHORT_MESSAGE });
    return;
  }
  if (jobDescription.length > MAX_JOB_DESCRIPTION_CHARS) {
    res.status(400).json({ ok: false, error: 'too_long', message: TOO_LONG_MESSAGE });
    return;
  }

  const ip = getClientIp(req);
  if (Math.random() < 0.05) pruneVisitorHits();
  if (isRateLimited(ip)) {
    res.status(429).json({ ok: false, error: 'rate_limited', message: RATE_LIMIT_MESSAGE });
    return;
  }

  if (!process.env.GROQ_API_KEY) {
    console.error('GROQ_API_KEY is not set');
    res.status(200).json({ ok: false, error: 'busy', message: BUSY_MESSAGE });
    return;
  }

  let digest;
  try {
    digest = buildProfileDigest(loadProfile());
  } catch (err) {
    console.error('Failed to load profile.json', err);
    res.status(200).json({ ok: false, error: 'busy', message: BUSY_MESSAGE });
    return;
  }

  const userMessage =
    `PROFILE (JSON):\n${JSON.stringify(digest)}\n\n` +
    `JOB DESCRIPTION PASTED BY THE RECRUITER:\n${jobDescription}`;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 25000);

    const groqRes = await fetch(GROQ_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        temperature: 0.4,
        max_tokens: 1200,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          { role: 'user', content: userMessage },
        ],
      }),
      signal: controller.signal,
    }).finally(() => clearTimeout(timeout));

    if (!groqRes.ok) {
      // Covers Groq being down, invalid key, and 429 (free-tier limit hit).
      console.error('Groq API error', groqRes.status, await groqRes.text().catch(() => ''));
      res.status(200).json({ ok: false, error: 'busy', message: BUSY_MESSAGE });
      return;
    }

    const data = await groqRes.json();
    const content = data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;

    const report = extractJson(content || '');
    if (!validateReport(report)) {
      console.error('Groq response failed schema validation', content);
      res.status(200).json({ ok: false, error: 'busy', message: BUSY_MESSAGE });
      return;
    }

    res.status(200).json({ ok: true, report });
  } catch (err) {
    console.error('Unexpected error calling Groq', err);
    res.status(200).json({ ok: false, error: 'busy', message: BUSY_MESSAGE });
  }
};
