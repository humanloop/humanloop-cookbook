import { distance } from "fastest-levenshtein";
import { Humanloop } from "humanloop";

/* Extracts answer from generation.
Handles a generation that if separated by "---" with the answer being the first part.
Also handles a generation that starts with "```\n" and removes it.
*/
function extractAnswer(generation: string) {
  let answer = generation.split("---")[0].trim();
  if (answer.startsWith("```\n")) {
    answer = answer.substring(4).trim();
  }

  return answer;
}

export function levenshtein(
  log: Humanloop.LogResponse,
  testcase: Humanloop.DatapointResponse
) {
  const target = testcase.target!.output as string;
  const generated = extractAnswer(log.output!);
  return distance(target, generated);
}
