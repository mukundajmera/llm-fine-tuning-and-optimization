import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { TrainingMetricsChart } from "@/components/training-metrics-chart";

interface Props {
  params: Promise<{ id: string }>;
}

async function getJob(id: string) {
  // In production, fetch from API
  return {
    id,
    job_name: "llama-8b-customer-support",
    base_model: "meta-llama/Llama-3.1-8B",
    dataset_path: "gs://my-bucket/data/train.jsonl",
    status: "completed",
    created_at: "2024-12-29T10:00:00Z",
    completed_at: "2024-12-29T14:00:00Z",
    duration_seconds: 14400,
    gpu_type: "A100-40GB",
    total_cost: 45.50,
    hyperparameters: {
      lora_r: 64,
      lora_alpha: 16,
      learning_rate: 0.0002,
      num_epochs: 3,
      batch_size: 4,
    },
    metrics: {
      final_train_loss: 0.45,
      final_eval_loss: 0.52,
      samples_per_second: 15.2,
    },
  };
}

export default async function JobDetailPage({ params }: Props) {
  const { id } = await params;
  const job = await getJob(id);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">{job.job_name}</h1>
          <p className="text-muted-foreground">{job.base_model}</p>
        </div>
        <span
          className={`px-3 py-1 rounded-full text-sm font-medium ${
            job.status === "completed"
              ? "bg-green-100 text-green-800"
              : job.status === "running"
              ? "bg-blue-100 text-blue-800"
              : "bg-gray-100 text-gray-800"
          }`}
        >
          {job.status}
        </span>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDescription>Training hyperparameters</CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="space-y-2">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">LoRA Rank</dt>
                <dd className="font-medium">{job.hyperparameters.lora_r}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">LoRA Alpha</dt>
                <dd className="font-medium">{job.hyperparameters.lora_alpha}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Learning Rate</dt>
                <dd className="font-medium">{job.hyperparameters.learning_rate}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Epochs</dt>
                <dd className="font-medium">{job.hyperparameters.num_epochs}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Batch Size</dt>
                <dd className="font-medium">{job.hyperparameters.batch_size}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">GPU Type</dt>
                <dd className="font-medium">{job.gpu_type}</dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Summary</CardTitle>
            <CardDescription>Training results</CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="space-y-2">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Duration</dt>
                <dd className="font-medium">
                  {Math.round(job.duration_seconds / 3600)} hours
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Final Train Loss</dt>
                <dd className="font-medium">{job.metrics.final_train_loss.toFixed(4)}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Final Eval Loss</dt>
                <dd className="font-medium">{job.metrics.final_eval_loss.toFixed(4)}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Throughput</dt>
                <dd className="font-medium">{job.metrics.samples_per_second.toFixed(1)} samples/sec</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Total Cost</dt>
                <dd className="font-medium text-green-600">${job.total_cost.toFixed(2)}</dd>
              </div>
            </dl>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Training Metrics</CardTitle>
          <CardDescription>Loss over training steps</CardDescription>
        </CardHeader>
        <CardContent>
          <TrainingMetricsChart />
        </CardContent>
      </Card>

      {job.status === "completed" && (
        <Card>
          <CardHeader>
            <CardTitle>Actions</CardTitle>
          </CardHeader>
          <CardContent className="flex gap-4">
            <a
              href={`/deployments/new?job_id=${job.id}`}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90"
            >
              Deploy Model
            </a>
            <button className="px-4 py-2 border border-input bg-background hover:bg-accent rounded-md text-sm font-medium">
              Download Model
            </button>
            <button className="px-4 py-2 border border-input bg-background hover:bg-accent rounded-md text-sm font-medium">
              Run Evaluation
            </button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
