import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function Home() {
  return (
    <div className="space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">
          Welcome to LLMForge
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Production LLM Fine-Tuning & Deployment Platform. Train custom LLMs with QLoRA and deploy with vLLM.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mt-12">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              🎓 Training
            </CardTitle>
            <CardDescription>
              Fine-tune LLMs with QLoRA
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm">
              <li>✅ Llama 3.1 8B/70B support</li>
              <li>✅ Mistral 7B v0.3 support</li>
              <li>✅ 4-bit NF4 quantization</li>
              <li>✅ MLflow tracking</li>
            </ul>
            <a
              href="/jobs/new"
              className="inline-block mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90"
            >
              Start Training
            </a>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              🚀 Deployment
            </CardTitle>
            <CardDescription>
              Deploy with vLLM on GKE
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm">
              <li>✅ vLLM inference engine</li>
              <li>✅ Autoscaling on GKE</li>
              <li>✅ L4/A100 GPU support</li>
              <li>✅ OpenAI-compatible API</li>
            </ul>
            <a
              href="/deployments"
              className="inline-block mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90"
            >
              View Deployments
            </a>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              💬 Inference
            </CardTitle>
            <CardDescription>
              Test your models
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm">
              <li>✅ Chat completions</li>
              <li>✅ Text completions</li>
              <li>✅ Streaming support</li>
              <li>✅ Cost tracking</li>
            </ul>
            <a
              href="/inference"
              className="inline-block mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90"
            >
              Try Inference
            </a>
          </CardContent>
        </Card>
      </div>

      <div className="mt-12 p-6 bg-muted rounded-lg">
        <h2 className="text-2xl font-bold mb-4">Quick Stats</h2>
        <div className="grid md:grid-cols-4 gap-4">
          <div className="bg-background p-4 rounded-md">
            <div className="text-3xl font-bold">0</div>
            <div className="text-sm text-muted-foreground">Training Jobs</div>
          </div>
          <div className="bg-background p-4 rounded-md">
            <div className="text-3xl font-bold">0</div>
            <div className="text-sm text-muted-foreground">Active Deployments</div>
          </div>
          <div className="bg-background p-4 rounded-md">
            <div className="text-3xl font-bold">0</div>
            <div className="text-sm text-muted-foreground">Inference Requests</div>
          </div>
          <div className="bg-background p-4 rounded-md">
            <div className="text-3xl font-bold">$0.00</div>
            <div className="text-sm text-muted-foreground">Total Cost</div>
          </div>
        </div>
      </div>
    </div>
  );
}
