#!/usr/bin/env node
/** Extract accepted teacher trajectories and both two-teacher consensus variants. */

const fs = require("fs");
const path = require("path");

const directory = __dirname;
const teachers = ["deepseek-v4-pro", "glm-5.2"];
const sources = [
  "rw_gen_coherence_4811_distill.jsonl",
  "rw_gen_positioning_check_2822_distill.jsonl",
  "rw_gen_positioning_type_954_distill.jsonl",
];

function derivedFilename(sourceFilename, suffix, sampleCount) {
  return sourceFilename.replace(
    /_\d+_distill\.jsonl$/,
    `_${sampleCount}_distill_${suffix}.jsonl`,
  );
}

function writeRows(filename, rows) {
  const outputPath = path.join(directory, filename);
  const temporaryPath = `${outputPath}.tmp`;
  fs.writeFileSync(temporaryPath, `${rows.map(JSON.stringify).join("\n")}\n`);
  fs.renameSync(temporaryPath, outputPath);

  const parts = filename.match(/^(.*)_\d+(_distill_.*\.jsonl)$/);
  if (parts === null) throw new Error(`Invalid derived filename: ${filename}`);
  const [, prefix, suffix] = parts;
  for (const candidate of fs.readdirSync(directory)) {
    if (
      candidate !== filename &&
      candidate.startsWith(`${prefix}_`) &&
      candidate.endsWith(suffix) &&
      /^\d+$/.test(candidate.slice(prefix.length + 1, -suffix.length))
    ) {
      fs.unlinkSync(path.join(directory, candidate));
    }
  }
  console.log(`${filename}: ${rows.length}`);
}

function validatePair(deepseek, glm, sourceFilename) {
  const deepseekPrompt = JSON.stringify(deepseek.messages.slice(0, -1));
  const glmPrompt = JSON.stringify(glm.messages.slice(0, -1));
  if (
    deepseek.teacher_label !== glm.teacher_label ||
    deepseek.gold_label !== glm.gold_label ||
    deepseek.task !== glm.task ||
    deepseek.aspect !== glm.aspect ||
    deepseekPrompt !== glmPrompt
  ) {
    throw new Error(`${sourceFilename}: incompatible teacher rows for id=${deepseek.id}`);
  }
}

function addConsensusMetadata(row, deepseek, glm) {
  return {
    ...row,
    consensus_models: teachers,
    consensus_teacher_labels: {
      [deepseek.teacher_model]: deepseek.teacher_label,
      [glm.teacher_model]: glm.teacher_label,
    },
    consensus_trajectory_teacher: row.teacher_model,
  };
}

for (const sourceFilename of sources) {
  const sourcePath = path.join(directory, sourceFilename);
  const rowsByTeacher = new Map(teachers.map((teacher) => [teacher, new Map()]));
  const lines = fs.readFileSync(sourcePath, "utf8").trimEnd().split("\n");

  lines.forEach((line, index) => {
    const row = JSON.parse(line);
    if (row.record_type !== "distillation" || row.accepted !== true) return;
    if (!rowsByTeacher.has(row.teacher_model)) return;
    if (row.teacher_label !== row.gold_label) {
      throw new Error(`${sourceFilename}:${index + 1} accepted an incorrect label`);
    }
    const teacherRows = rowsByTeacher.get(row.teacher_model);
    if (teacherRows.has(row.id)) {
      throw new Error(
        `${sourceFilename}:${index + 1} duplicates ${row.teacher_model}/${row.id}`,
      );
    }
    teacherRows.set(row.id, row);
  });

  for (const teacher of teachers) {
    writeRows(
      derivedFilename(sourceFilename, teacher, rowsByTeacher.get(teacher).size),
      [...rowsByTeacher.get(teacher).values()],
    );
  }

  const [deepseekTeacher, glmTeacher] = teachers;
  const deepseekConsensus = [];
  const glmConsensus = [];
  for (const [id, deepseek] of rowsByTeacher.get(deepseekTeacher)) {
    const glm = rowsByTeacher.get(glmTeacher).get(id);
    if (glm === undefined) continue;
    validatePair(deepseek, glm, sourceFilename);
    deepseekConsensus.push(addConsensusMetadata(deepseek, deepseek, glm));
    glmConsensus.push(addConsensusMetadata(glm, deepseek, glm));
  }

  const consensusPrefix = `${teachers.join("_")}_consensus`;
  writeRows(
    derivedFilename(
      sourceFilename,
      `${consensusPrefix}_${deepseekTeacher}`,
      deepseekConsensus.length,
    ),
    deepseekConsensus,
  );
  writeRows(
    derivedFilename(
      sourceFilename,
      `${consensusPrefix}_${glmTeacher}`,
      glmConsensus.length,
    ),
    glmConsensus,
  );
}
