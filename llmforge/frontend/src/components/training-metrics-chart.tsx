"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

// Sample training metrics data
const data = [
  { step: 0, train_loss: 2.5, eval_loss: 2.6 },
  { step: 100, train_loss: 1.8, eval_loss: 1.9 },
  { step: 200, train_loss: 1.2, eval_loss: 1.4 },
  { step: 300, train_loss: 0.9, eval_loss: 1.1 },
  { step: 400, train_loss: 0.7, eval_loss: 0.9 },
  { step: 500, train_loss: 0.6, eval_loss: 0.75 },
  { step: 600, train_loss: 0.5, eval_loss: 0.65 },
  { step: 700, train_loss: 0.48, eval_loss: 0.58 },
  { step: 800, train_loss: 0.46, eval_loss: 0.54 },
  { step: 900, train_loss: 0.45, eval_loss: 0.52 },
];

export function TrainingMetricsChart() {
  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="step"
            label={{ value: "Training Steps", position: "bottom", offset: 0 }}
          />
          <YAxis
            label={{ value: "Loss", angle: -90, position: "insideLeft" }}
          />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="train_loss"
            stroke="#8884d8"
            name="Training Loss"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="eval_loss"
            stroke="#82ca9d"
            name="Evaluation Loss"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
