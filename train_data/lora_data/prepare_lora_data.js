#!/usr/bin/env node
/** Convert accepted distillation datasets into TRL prompt-completion JSONL. */

const fs = require("fs");
const path = require("path");

const outputDirectory = __dirname;
const distillDirectory = path.resolve(outputDirectory, "../distill_data");
const teachers = ["deepseek-v4-pro", "glm-5.2"];
const datasetPrefixes = [
  "rw_gen_coherence",
  "rw_gen_positioning_check",
  "rw_gen_positioning_type",
];
const singleTeacherDatasets = [
  {
    prefix: "rev_util_actionability",
    suffix: "deepseek-v4-pro",
    teacherModels: ["deepseek-v4-pro"],
    trajectoryTeacher: "deepseek-v4-pro",
  },
];
const variants = [
  {
    suffix: "deepseek-v4-pro",
    teacherModels: ["deepseek-v4-pro"],
    trajectoryTeacher: "deepseek-v4-pro",
  },
  {
    suffix: "glm-5.2",
    teacherModels: ["glm-5.2"],
    trajectoryTeacher: "glm-5.2",
  },
  {
    suffix: "deepseek-v4-pro_glm-5.2_consensus_deepseek-v4-pro",
    teacherModels: teachers,
    trajectoryTeacher: "deepseek-v4-pro",
  },
  {
    suffix: "deepseek-v4-pro_glm-5.2_consensus_glm-5.2",
    teacherModels: teachers,
    trajectoryTeacher: "glm-5.2",
  },
];
const exportRecords = [];

function findSourceFilename(prefix, variant) {
  const suffix = `_distill_${variant.suffix}.jsonl`;
  const matches = fs
    .readdirSync(distillDirectory)
    .filter((filename) => filename.startsWith(`${prefix}_`) && filename.endsWith(suffix));
  if (matches.length !== 1) {
    throw new Error(
      `Expected one ${prefix} dataset ending in ${suffix}, found: ${matches.join(", ") || "none"}`,
    );
  }
  return matches[0];
}

function convertRow(row, filename, lineNumber, variant) {
  if (
    row.record_type !== "distillation" ||
    row.accepted !== true ||
    row.teacher_label !== row.gold_label ||
    row.teacher_model !== variant.trajectoryTeacher
  ) {
    throw new Error(`${filename}:${lineNumber} is not the expected accepted trajectory`);
  }
  if (!Array.isArray(row.messages) || row.messages.length < 2) {
    throw new Error(`${filename}:${lineNumber} has no valid messages`);
  }
  if (
    variant.teacherModels.length === 2 &&
    (JSON.stringify(row.consensus_models) !== JSON.stringify(teachers) ||
      row.consensus_trajectory_teacher !== variant.trajectoryTeacher)
  ) {
    throw new Error(`${filename}:${lineNumber} has invalid consensus metadata`);
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
    teacher_models: variant.teacherModels,
    prompt,
    completion: [assistant],
  };
}

function writeRows(filename, rows, sourceFilename, sourceBytes, variant) {
  if (new Set(rows.map((row) => row.id)).size !== rows.length) {
    throw new Error(`${filename} contains duplicate sample IDs`);
  }
  const outputPath = path.join(outputDirectory, filename);
  const temporaryPath = `${outputPath}.tmp`;
  fs.writeFileSync(temporaryPath, `${rows.map(JSON.stringify).join("\n")}\n`);
  fs.renameSync(temporaryPath, outputPath);

  const labelCounts = {};
  for (const row of rows) labelCounts[row.label] = (labelCounts[row.label] || 0) + 1;
  exportRecords.push({
    file: filename,
    source_file: sourceFilename,
    source_bytes_read: sourceBytes,
    samples: rows.length,
    labels: labelCounts,
    teacher_models: variant.teacherModels,
    trajectory_teacher: variant.trajectoryTeacher,
  });
  console.log(`${filename}: ${rows.length}`);
}

for (const prefix of datasetPrefixes) {
  for (const variant of variants) {
    const filename = findSourceFilename(prefix, variant);
    const sourcePath = path.join(distillDirectory, filename);
    const content = fs.readFileSync(sourcePath, "utf8");
    const rows = content
      .trimEnd()
      .split("\n")
      .map((line, index) =>
        convertRow(JSON.parse(line), filename, index + 1, variant),
      );
    const sampleCount = Number(filename.slice(prefix.length + 1).split("_", 1)[0]);
    if (!Number.isInteger(sampleCount) || sampleCount !== rows.length) {
      throw new Error(
        `${filename} declares ${sampleCount} samples but contains ${rows.length}`,
      );
    }
    writeRows(filename, rows, filename, Buffer.byteLength(content), variant);
  }
}

for (const dataset of singleTeacherDatasets) {
  const variant = dataset;
  const filename = findSourceFilename(dataset.prefix, variant);
  const sourcePath = path.join(distillDirectory, filename);
  const content = fs.readFileSync(sourcePath, "utf8");
  const rows = content
    .trimEnd()
    .split("\n")
    .map((line, index) =>
      convertRow(JSON.parse(line), filename, index + 1, variant),
    );
  const sampleCount = Number(filename.slice(dataset.prefix.length + 1).split("_", 1)[0]);
  if (!Number.isInteger(sampleCount) || sampleCount !== rows.length) {
    throw new Error(
      `${filename} declares ${sampleCount} samples but contains ${rows.length}`,
    );
  }
  writeRows(filename, rows, filename, Buffer.byteLength(content), variant);
}

const manifestPath = path.join(outputDirectory, "export_manifest.json");
const temporaryManifestPath = `${manifestPath}.tmp`;
fs.writeFileSync(
  temporaryManifestPath,
  `${JSON.stringify(
    {
      schema_version: 2,
      generated_at_utc: new Date().toISOString(),
      outputs: exportRecords,
    },
    null,
    2,
  )}\n`,
);
fs.renameSync(temporaryManifestPath, manifestPath);
console.log(`export_manifest.json: ${exportRecords.length} outputs`);
