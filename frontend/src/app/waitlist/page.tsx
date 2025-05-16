import { WaitlistForm } from "@/components/waitlist/waitlist-form";

export default function WaitlistPage() {
  return (
    <div className="container mx-auto py-12 px-4">
      <div className="max-w-2xl mx-auto text-center">
        <h1 className="text-4xl font-bold tracking-tight mb-4">
          Join the Waitlist
        </h1>
        <p className="text-lg text-muted-foreground mb-8">
          Be the first to know when we launch. No spam, ever.
        </p>
        <WaitlistForm />
      </div>
    </div>
  );
} 