"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface InferenceResult {
  id: string;
  generated_text: string;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  cost_usd: number;
}

export function InferenceTester() {
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Mock response for demo
      await new Promise((resolve) => setTimeout(resolve, 1000));

      setResult({
        id: `cmpl-${Date.now()}`,
        generated_text: `This is a mock response to your prompt: "${prompt.slice(0, 50)}..."

The model would generate relevant content based on your fine-tuned training data. In a production deployment, this would be the actual model output from vLLM.

Key points:
1. Your prompt was processed successfully
2. The model is running on the configured GPU
3. Response was generated using the specified parameters

Temperature: ${temperature}
Max tokens: ${maxTokens}`,
        prompt_tokens: prompt.split(" ").length,
        completion_tokens: 75,
        latency_ms: 850,
        cost_usd: 0.0015,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate response");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid md:grid-cols-2 gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Input</CardTitle>
          <CardDescription>Configure your inference request</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="deployment">Deployment</Label>
              <Select defaultValue="demo">
                <SelectTrigger>
                  <SelectValue placeholder="Select deployment" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="demo">llama-8b-prod (Demo)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="prompt">Prompt</Label>
              <textarea
                id="prompt"
                className="flex min-h-[200px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                placeholder="Enter your prompt here..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="maxTokens">Max Tokens</Label>
                <Input
                  id="maxTokens"
                  type="number"
                  min={1}
                  max={4096}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="temperature">Temperature</Label>
                <Input
                  id="temperature"
                  type="number"
                  min={0}
                  max={2}
                  step={0.1}
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                />
              </div>
            </div>

            <Button type="submit" className="w-full" disabled={loading || !prompt.trim()}>
              {loading ? "Generating..." : "Generate Response"}
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Output</CardTitle>
          <CardDescription>Model response</CardDescription>
        </CardHeader>
        <CardContent>
          {error && (
            <div className="p-4 bg-destructive/10 text-destructive rounded-md">
              {error}
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <div className="p-4 bg-muted rounded-md">
                <pre className="whitespace-pre-wrap text-sm">
                  {result.generated_text}
                </pre>
              </div>

              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="flex justify-between p-2 bg-background rounded">
                  <span className="text-muted-foreground">Prompt Tokens</span>
                  <span className="font-medium">{result.prompt_tokens}</span>
                </div>
                <div className="flex justify-between p-2 bg-background rounded">
                  <span className="text-muted-foreground">Completion Tokens</span>
                  <span className="font-medium">{result.completion_tokens}</span>
                </div>
                <div className="flex justify-between p-2 bg-background rounded">
                  <span className="text-muted-foreground">Latency</span>
                  <span className="font-medium">{result.latency_ms}ms</span>
                </div>
                <div className="flex justify-between p-2 bg-background rounded">
                  <span className="text-muted-foreground">Cost</span>
                  <span className="font-medium text-green-600">
                    ${result.cost_usd.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>
          )}

          {!result && !error && (
            <div className="flex items-center justify-center h-[300px] text-muted-foreground">
              Enter a prompt and click Generate to see the response
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
