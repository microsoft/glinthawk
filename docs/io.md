# Understanding Glinthawk input/output

## Input JSON Schema

```json
{
  "metadata": {
    // Metadata about the input
  },
  "prompts": [
    {
      "id": "unique_id", // optional, auto-generated if not provided
      "temprature": 0.5, // optional, default is 0
      "output_len": 1024,

      // at least one of the following must be provided
      "prompt": "What is your name?",
      "prompt_tokens": [1, 2, 3, 4, 5],
    },
    { // bare minimum
      "text": "What is your favorite color?",
    }
  ],
}
```

## Output JSON Schema

```json
{
  "metadata": {
    // Metadata about the output
  },
  "completed_prompts": [
    {
      "id": "unique_id",
      "temprature": 0.5,
      "output_len": 1024,

      "prompt": "What is your name?",
      "prompt_tokens": [1, 2, 3, 4, 5],

      "completion": "My name is John Doe.",
      "completion_tokens": [6, 7, 8, 9, 10],
    },
    ...
  ],
}
```
