#!/usr/bin/env node
/** Convert accepted distillation logs into TRL conversational prompt-completion data. */

const fs = require("fs");
const path = require("path");

const outputDirectory = __dirname;
const distillDirectory = path.resolve(outputDirectory, "../distill_data");
const filenames = [
  "rw_gen_coherence_4811_distill_deepseek-v4-pro.jsonl",
  "rw_gen_coherence_4811_distill_glm-5.2.jsonl",
  "rw_gen_coherence_4811_distill_deepseek-v4-pro_glm-5.2_consensus.jsonl",
];

function teacherModels(row) {
  if (Array.isArray(row.consensus_models)) return row.consensus_models;
  return [row.teacher_model];
}

function convertRow(row, filename, lineNumber) {
  if (
    row.record_type !== "distillation" ||
    row.accepted !== true ||
    row.teacher_label !== row.gold_label
  ) {
    throw new Error(`${filename}:${lineNumber} is not an accepted correct trajectory`);
  }
  if (!Array.isArray(row.messages) || row.messages.length < 2) {
    throw new Error(`${filename}:${lineNumber} has no valid messages`);
  }

  const prompt = row.messages.slice(0, -1);
  const assistant = row.messages.at(-1);
  if (
    assistant?.role !== "assistant" ||
    typeof assistant.content !== "string" ||
    assistant.content !== row.completion
  ) {
    throw new Error(`${filename}:${lineNumber} has an invalid assistant completion`);
  }

  return {
    id: row.id,
    task: row.task,
    aspect: row.aspect,
    label: row.gold_label,
    teacher_models: teacherModels(row),
    prompt,
    completion: [assistant],
  };
}

for (const filename of filenames) {
  const inputPath = path.join(distillDirectory, filename);
  const outputPath = path.join(outputDirectory, filename);
  const rows = fs
    .readFileSync(inputPath, "utf8")
    .trimEnd()
    .split("\n")
    .map((line, index) => convertRow(JSON.parse(line), filename, index + 1));

  if (new Set(rows.map((row) => row.id)).size !== rows.length) {
    throw new Error(`${filename} contains duplicate sample IDs`);
  }

  const temporaryPath = `${outputPath}.tmp`;
  fs.writeFileSync(temporaryPath, `${rows.map(JSON.stringify).join("\n")}\n`);
  fs.renameSync(temporaryPath, outputPath);
  console.log(`${filename}: ${rows.length}`);
}
