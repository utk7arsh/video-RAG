import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const { email } = await req.json();
    
    const res = await fetch(`${BACKEND_URL}/waitlist`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });
    
    const data = await res.json();
    
    if (!res.ok) {
      return NextResponse.json(
        { detail: data.detail || "Error joining waitlist" },
        { status: res.status }
      );
    }
    
    return NextResponse.json(data);
  } catch (error) {
    console.error("Waitlist error:", error);
    return NextResponse.json(
      { detail: "Internal server error" },
      { status: 500 }
    );
  }
} 