import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "LLMForge - LLM Fine-Tuning Platform",
  description: "Production LLM Fine-Tuning & Deployment Platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-background">
          <nav className="border-b">
            <div className="container mx-auto px-4 py-4">
              <div className="flex items-center justify-between">
                <a href="/" className="text-xl font-bold">
                  🔥 LLMForge
                </a>
                <div className="flex items-center space-x-6">
                  <a href="/jobs" className="text-sm font-medium hover:text-primary">
                    Training Jobs
                  </a>
                  <a href="/deployments" className="text-sm font-medium hover:text-primary">
                    Deployments
                  </a>
                  <a href="/inference" className="text-sm font-medium hover:text-primary">
                    Inference
                  </a>
                </div>
              </div>
            </div>
          </nav>
          <main className="container mx-auto px-4 py-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
