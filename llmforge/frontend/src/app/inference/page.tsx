import { InferenceTester } from "@/components/inference-tester";

export default function InferencePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Inference Playground</h1>
        <p className="text-muted-foreground">
          Test your deployed models
        </p>
      </div>

      <InferenceTester />
    </div>
  );
}
