import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface Job {
  id: string;
  job_name: string;
  base_model: string;
  status: string;
  created_at: string;
  total_cost?: number;
}

async function getJobs(): Promise<Job[]> {
  // In production, this would fetch from the API
  // For now, return mock data
  return [
    {
      id: "1",
      job_name: "llama-8b-customer-support",
      base_model: "meta-llama/Llama-3.1-8B",
      status: "completed",
      created_at: "2024-12-29T10:00:00Z",
      total_cost: 45.50,
    },
    {
      id: "2",
      job_name: "mistral-7b-code-assistant",
      base_model: "mistralai/Mistral-7B-v0.3",
      status: "running",
      created_at: "2024-12-29T08:00:00Z",
    },
  ];
}

function getStatusColor(status: string): string {
  switch (status) {
    case "completed":
      return "bg-green-100 text-green-800";
    case "running":
      return "bg-blue-100 text-blue-800";
    case "failed":
      return "bg-red-100 text-red-800";
    case "queued":
      return "bg-yellow-100 text-yellow-800";
    default:
      return "bg-gray-100 text-gray-800";
  }
}

export default async function JobsPage() {
  const jobs = await getJobs();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Training Jobs</h1>
          <p className="text-muted-foreground">
            Manage your LLM fine-tuning jobs
          </p>
        </div>
        <a
          href="/jobs/new"
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90"
        >
          + New Training Job
        </a>
      </div>

      <div className="grid gap-4">
        {jobs.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground mb-4">
                No training jobs yet. Create your first one!
              </p>
              <a
                href="/jobs/new"
                className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90"
              >
                Create Training Job
              </a>
            </CardContent>
          </Card>
        ) : (
          jobs.map((job) => (
            <Card key={job.id}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>{job.job_name}</CardTitle>
                  <span
                    className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(
                      job.status
                    )}`}
                  >
                    {job.status}
                  </span>
                </div>
                <CardDescription>{job.base_model}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between text-sm">
                  <div>
                    <span className="text-muted-foreground">Created: </span>
                    {new Date(job.created_at).toLocaleDateString()}
                  </div>
                  {job.total_cost && (
                    <div>
                      <span className="text-muted-foreground">Cost: </span>
                      ${job.total_cost.toFixed(2)}
                    </div>
                  )}
                  <a
                    href={`/jobs/${job.id}`}
                    className="text-primary hover:underline"
                  >
                    View Details →
                  </a>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}
