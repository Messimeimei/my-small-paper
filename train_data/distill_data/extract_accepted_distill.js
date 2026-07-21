#!/usr/bin/env node
/** Extract accepted teacher trajectories and a two-teacher consensus set. */

const fs = require("fs");
const path = require("path");

const directory = __dirname;
const source = path.join(directory, "rw_gen_coherence_4811_distill.jsonl");
const teachers = ["deepseek-v4-pro", "glm-5.2"];
const outputs = new Map(
  teachers.map((teacher) => [
    teacher,
    path.join(directory, `rw_gen_coherence_4811_distill_${teacher}.jsonl`),
  ]),
);
const consensusOutput = path.join(
  directory,
  "rw_gen_coherence_4811_distill_deepseek-v4-pro_glm-5.2_consensus.jsonl",
);

const rowsByTeacher = new Map(teachers.map((teacher) => [teacher, new Map()]));
for (const [lineNumber, line] of fs
  .readFileSync(source, "utf8")
  .trimEnd()
  .split("\n")
  .entries()) {
  const row = JSON.parse(line);
  if (row.record_type !== "distillation" || row.accepted !== true) continue;
  if (!rowsByTeacher.has(row.teacher_model)) continue;
  rowsByTeacher.get(row.teacher_model).set(row.id, row);
}

for (const teacher of teachers) {
  const rows = [...rowsByTeacher.get(teacher).values()];
  fs.writeFileSync(outputs.get(teacher), `${rows.map(JSON.stringify).join("\n")}\n`);
}

const [primaryTeacher, secondaryTeacher] = teachers;
const consensusRows = [];
for (const [id, primary] of rowsByTeacher.get(primaryTeacher)) {
  const secondary = rowsByTeacher.get(secondaryTeacher).get(id);
  if (!secondary || primary.teacher_label !== secondary.teacher_label) continue;
  consensusRows.push({
    ...primary,
    consensus_models: teachers,
    consensus_teacher_labels: {
      [primaryTeacher]: primary.teacher_label,
      [secondaryTeacher]: secondary.teacher_label,
    },
  });
}
fs.writeFileSync(consensusOutput, `${consensusRows.map(JSON.stringify).join("\n")}\n`);

console.log(`${primaryTeacher}: ${rowsByTeacher.get(primaryTeacher).size}`);
console.log(`${secondaryTeacher}: ${rowsByTeacher.get(secondaryTeacher).size}`);
console.log(`consensus: ${consensusRows.length}`);
