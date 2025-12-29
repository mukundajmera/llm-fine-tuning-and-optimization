import { JobConfigForm } from "@/components/job-config-form";

export default function NewJobPage() {
  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Create Training Job</h1>
        <p className="text-muted-foreground">
          Configure and launch a new QLoRA fine-tuning job
        </p>
      </div>

      <JobConfigForm />
    </div>
  );
}
