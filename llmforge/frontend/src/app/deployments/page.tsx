import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface Deployment {
  id: string;
  deployment_name: string;
  model_path: string;
  status: string;
  endpoint_url?: string;
  replicas: number;
  created_at: string;
}

async function getDeployments(): Promise<Deployment[]> {
  // In production, fetch from API
  return [
    {
      id: "1",
      deployment_name: "llama-8b-prod",
      model_path: "gs://my-bucket/models/llama-8b-ft",
      status: "active",
      endpoint_url: "https://api.llmforge.example.com/v1",
      replicas: 2,
      created_at: "2024-12-29T15:00:00Z",
    },
  ];
}

function getStatusColor(status: string): string {
  switch (status) {
    case "active":
      return "bg-green-100 text-green-800";
    case "deploying":
      return "bg-blue-100 text-blue-800";
    case "inactive":
      return "bg-gray-100 text-gray-800";
    default:
      return "bg-gray-100 text-gray-800";
  }
}

export default async function DeploymentsPage() {
  const deployments = await getDeployments();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Deployments</h1>
          <p className="text-muted-foreground">
            Manage your model deployments
          </p>
        </div>
      </div>

      <div className="grid gap-4">
        {deployments.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground mb-4">
                No deployments yet. Complete a training job first!
              </p>
              <a
                href="/jobs"
                className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90"
              >
                View Training Jobs
              </a>
            </CardContent>
          </Card>
        ) : (
          deployments.map((deployment) => (
            <Card key={deployment.id}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>{deployment.deployment_name}</CardTitle>
                  <span
                    className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(
                      deployment.status
                    )}`}
                  >
                    {deployment.status}
                  </span>
                </div>
                <CardDescription>
                  {deployment.replicas} replica(s) • Created {new Date(deployment.created_at).toLocaleDateString()}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {deployment.endpoint_url && (
                    <div className="flex items-center gap-2">
                      <span className="text-muted-foreground text-sm">Endpoint:</span>
                      <code className="text-sm bg-muted px-2 py-1 rounded">
                        {deployment.endpoint_url}
                      </code>
                    </div>
                  )}
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground text-sm">Model:</span>
                    <code className="text-sm bg-muted px-2 py-1 rounded">
                      {deployment.model_path}
                    </code>
                  </div>
                </div>
                <div className="mt-4 flex gap-2">
                  <button className="px-3 py-1 bg-primary text-primary-foreground rounded text-sm">
                    Scale
                  </button>
                  <button className="px-3 py-1 border border-input rounded text-sm">
                    Logs
                  </button>
                  <button className="px-3 py-1 border border-destructive text-destructive rounded text-sm">
                    Delete
                  </button>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}
