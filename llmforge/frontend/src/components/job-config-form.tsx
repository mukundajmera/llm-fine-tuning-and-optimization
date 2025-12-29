"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const formSchema = z.object({
  jobName: z.string().min(3, "Job name must be at least 3 characters").max(50),
  baseModel: z.enum([
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "mistralai/Mistral-7B-v0.3",
  ]),
  datasetPath: z.string().startsWith("gs://", "Dataset path must be a GCS path (gs://)"),
  loraR: z.coerce.number().int().min(8).max(128),
  loraAlpha: z.coerce.number().int().min(8).max(128),
  learningRate: z.coerce.number().min(0.00001).max(0.001),
  numEpochs: z.coerce.number().int().min(1).max(10),
  batchSize: z.coerce.number().int().min(1).max(32),
  maxSeqLength: z.coerce.number().int().min(256).max(8192),
  gpuType: z.enum(["A100-40GB", "A100-80GB"]),
});

type FormData = z.infer<typeof formSchema>;

export function JobConfigForm() {
  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors, isSubmitting },
  } = useForm<FormData>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      loraR: 64,
      loraAlpha: 16,
      learningRate: 0.0002,
      numEpochs: 3,
      batchSize: 4,
      maxSeqLength: 2048,
      gpuType: "A100-40GB",
      baseModel: "meta-llama/Llama-3.1-8B",
    },
  });

  const onSubmit = async (values: FormData) => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/jobs`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            job_name: values.jobName,
            base_model: values.baseModel,
            dataset_path: values.datasetPath,
            hyperparameters: {
              lora_r: values.loraR,
              lora_alpha: values.loraAlpha,
              learning_rate: values.learningRate,
              num_epochs: values.numEpochs,
              batch_size: values.batchSize,
              gradient_accumulation_steps: 4,
              max_seq_length: values.maxSeqLength,
            },
            gpu_type: values.gpuType,
          }),
        }
      );

      if (response.ok) {
        const job = await response.json();
        window.location.href = `/jobs/${job.id}`;
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail || "Failed to create job"}`);
      }
    } catch (error) {
      console.error("Failed to create job:", error);
      alert("Failed to create training job. Check console for details.");
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Basic Configuration</CardTitle>
          <CardDescription>Job name, model, and dataset</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="jobName">Job Name</Label>
            <Input
              id="jobName"
              placeholder="my-llama-finetuning"
              {...register("jobName")}
            />
            {errors.jobName && (
              <p className="text-sm text-destructive">{errors.jobName.message}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="baseModel">Base Model</Label>
            <Select
              defaultValue="meta-llama/Llama-3.1-8B"
              onValueChange={(value) => setValue("baseModel", value as FormData["baseModel"])}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select base model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="meta-llama/Llama-3.1-8B">
                  Llama 3.1 8B
                </SelectItem>
                <SelectItem value="meta-llama/Llama-3.1-70B">
                  Llama 3.1 70B
                </SelectItem>
                <SelectItem value="mistralai/Mistral-7B-v0.3">
                  Mistral 7B v0.3
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="datasetPath">Dataset Path (GCS)</Label>
            <Input
              id="datasetPath"
              placeholder="gs://your-bucket/data/train.jsonl"
              {...register("datasetPath")}
            />
            {errors.datasetPath && (
              <p className="text-sm text-destructive">{errors.datasetPath.message}</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>LoRA Configuration</CardTitle>
          <CardDescription>Low-Rank Adaptation hyperparameters</CardDescription>
        </CardHeader>
        <CardContent className="grid md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="loraR">LoRA Rank (r)</Label>
            <Input
              id="loraR"
              type="number"
              {...register("loraR")}
            />
            <p className="text-xs text-muted-foreground">Typical: 8-128, higher = more capacity</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="loraAlpha">LoRA Alpha</Label>
            <Input
              id="loraAlpha"
              type="number"
              {...register("loraAlpha")}
            />
            <p className="text-xs text-muted-foreground">Scaling factor, typically r/4 to r*2</p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Training Hyperparameters</CardTitle>
          <CardDescription>Optimizer and training settings</CardDescription>
        </CardHeader>
        <CardContent className="grid md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="learningRate">Learning Rate</Label>
            <Input
              id="learningRate"
              type="number"
              step="0.00001"
              {...register("learningRate")}
            />
            <p className="text-xs text-muted-foreground">Recommended: 2e-4</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="numEpochs">Number of Epochs</Label>
            <Input
              id="numEpochs"
              type="number"
              {...register("numEpochs")}
            />
            <p className="text-xs text-muted-foreground">Typical: 1-5</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="batchSize">Batch Size</Label>
            <Input
              id="batchSize"
              type="number"
              {...register("batchSize")}
            />
            <p className="text-xs text-muted-foreground">Per GPU batch size</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="maxSeqLength">Max Sequence Length</Label>
            <Input
              id="maxSeqLength"
              type="number"
              {...register("maxSeqLength")}
            />
            <p className="text-xs text-muted-foreground">Context window for training</p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Compute Configuration</CardTitle>
          <CardDescription>GPU and infrastructure settings</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Label htmlFor="gpuType">GPU Type</Label>
            <Select
              defaultValue="A100-40GB"
              onValueChange={(value) => setValue("gpuType", value as FormData["gpuType"])}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select GPU" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="A100-40GB">
                  A100 40GB ($3.67/hr)
                </SelectItem>
                <SelectItem value="A100-80GB">
                  A100 80GB ($5.24/hr)
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <div className="flex gap-4">
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Creating..." : "Create Training Job"}
        </Button>
        <Button type="button" variant="outline" onClick={() => window.history.back()}>
          Cancel
        </Button>
      </div>
    </form>
  );
}
