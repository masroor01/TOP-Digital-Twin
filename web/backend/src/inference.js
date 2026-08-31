// Thin re-export: the actual inference now runs locally in-process via
// pure-JS model files (see src/models/index.js) -- no Python service, no
// network hop. Kept as a separate module so routes.js's import doesn't
// need to change if the inference backend ever changes again.
export { predictBatch, inferenceHealth } from './models/index.js';
