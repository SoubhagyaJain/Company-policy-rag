import React from "react";
import { ThinkingPanel } from "./ThinkingPanel";
import { ThinkingEvent } from "../types/thinking";

interface Props {
  thinking?: string;
  events?: ThinkingEvent[];
}

export function ThinkingBlock({ thinking, events }: Props) {
  if (events && events.length > 0) {
    return <ThinkingPanel events={events} />;
  }

  if (!thinking || !thinking.trim()) return null;

  const syntheticEvent: ThinkingEvent = {
    id: "legacy_thk",
    query_id: "",
    stage: "answer_planning",
    status: "completed",
    title: "Reasoning",
    summary: thinking,
    duration_ms: 0,
  };

  return <ThinkingPanel events={[syntheticEvent]} />;
}