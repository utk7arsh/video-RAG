import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useTheme } from '@/contexts/ThemeContext';
import { Navbar } from '@/components/layout/Navbar';
import { Footer } from '@/components/layout/Footer';
import { Button } from '@/components/ui/button';
import { ArrowRight, Play, Search, Book } from 'lucide-react';

// Get API URL from environment variables with fallback
const API_URL = import.meta.env.VITE_API_URL;
console.log('API URL:', API_URL); // Debug log

const Index = () => {
  const { toggleTheme, isDarkMode } = useTheme();
  const [email, setEmail] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState('');
  const [showWaitlistModal, setShowWaitlistModal] = useState(false);

  async function handleWaitlistSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    console.log('Form submitted with email:', email);
    
    if (!email.match(/^[^@\s]+@[^@\s]+\.[^@\s]+$/)) {
      setError('Please enter a valid email address.');
      return;
    }
    
    if (!API_URL) {
      console.error('API URL is not configured:', import.meta.env);
      setError('Service temporarily unavailable. Please try again later.');
      return;
    }
    
    try {
      console.log('Making API request to:', `${API_URL}/waitlist`);
      const res = await fetch(`${API_URL}/waitlist`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        mode: 'cors',
        credentials: 'omit',
        body: JSON.stringify({ email }),
      });
      
      console.log('Response status:', res.status);
      const data = await res.json();
      console.log('Response data:', data);
      
      if (res.ok) {
        setSubmitted(true);
        setEmail('');
        setShowWaitlistModal(false);
      } else {
        setError(data.detail || 'Something went wrong. Please try again.');
      }
    } catch (err) {
      console.error('Waitlist error:', err);
      setError('Service temporarily unavailable. Please try again later.');
    }
  }

  function handleOpenWaitlistModal() {
    setShowWaitlistModal(true);
    setSubmitted(false);
    setEmail('');
    setError('');
  }

  function handleCloseWaitlistModal() {
    setShowWaitlistModal(false);
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar toggleTheme={toggleTheme} isDarkMode={isDarkMode} />
      
      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4">
        <div className="container mx-auto">
          <div className="flex flex-col md:flex-row items-center">
            <div className="md:w-1/2 space-y-6">
              <h1 className="text-4xl md:text-6xl font-bold leading-tight">
                Extract <span className="gradient-text">Insights</span> from Video Content with AI
              </h1>
              <p className="text-xl text-muted-foreground">
                Turn educational videos into interactive learning experiences. Ask questions, get instant answers, and save key insights.
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                {/* <Link to="/signup">
                  <Button size="lg" className="gradient-bg w-full sm:w-auto">
                    Get Started <ArrowRight className="ml-2 h-5 w-5" />
                  </Button>
                </Link> */}
                <Button size="lg" className="gradient-bg w-full sm:w-auto" onClick={handleOpenWaitlistModal}>
                  Get Started <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
                <Link to="/how-it-works">
                  <Button size="lg" variant="outline" className="w-full sm:w-auto">
                    How It Works
                  </Button>
                </Link>
              </div>
            </div>
            
            <div className="md:w-1/2 mt-10 md:mt-0 flex justify-center">
              {/* Video Player Preview */}
              <div className="relative shadow-2xl rounded-xl overflow-hidden w-full max-w-lg animate-fade-in">
                <div className="aspect-video bg-muted/30 flex items-center justify-center relative">
                  <div className="absolute top-0 left-0 right-0 h-16 bg-gradient-to-b from-black/40 to-transparent"></div>
                  <div className="w-16 h-16 rounded-full gradient-bg flex items-center justify-center shadow-lg animate-pulse-glow cursor-pointer">
                    <Play className="h-8 w-8 text-white" fill="white" />
                  </div>
                  <img 
                    src="https://images.unsplash.com/photo-1605711285791-0219e80e43a3" 
                    alt="Video Thumbnail" 
                    className="absolute inset-0 w-full h-full object-cover z-[-1]"
                  />
                </div>
                
                {/* Chat overlay */}
                <div className="absolute right-4 bottom-4 bg-background/90 backdrop-blur-md p-4 rounded-lg border border-border w-48 shadow-lg">
                  <div className="text-xs font-medium mb-2">AI Analysis</div>
                  <div className="space-y-2">
                    <div className="text-xs bg-primary/10 p-2 rounded">What's the key point at 2:30?</div>
                    <div className="text-xs bg-muted p-2 rounded">The speaker explains how neural networks process image data.</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* Features Section */}
      <section className="py-20 px-4 bg-muted/30">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold">
              Powerful Video Analysis <span className="gradient-text">Features</span>
            </h2>
            <p className="text-xl text-muted-foreground mt-4 max-w-3xl mx-auto">
              Extract maximum value from educational content with our AI-powered video analysis tools.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="bg-background rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-border hover-scale">
              <div className="w-12 h-12 rounded-lg gradient-bg flex items-center justify-center mb-4">
                <Search className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Intelligent Video Search</h3>
              <p className="text-muted-foreground">
                Ask questions about any moment in your videos and get precise answers with timestamp references.
              </p>
            </div>
            
            {/* Feature 2 */}
            <div className="bg-background rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-border hover-scale">
              <div className="w-12 h-12 rounded-lg gradient-bg flex items-center justify-center mb-4">
                <Book className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Automatic Insights</h3>
              <p className="text-muted-foreground">
                Our AI automatically identifies key concepts, summaries, and important moments from your videos.
              </p>
            </div>
            
            {/* Feature 3 */}
            <div className="bg-background rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-border hover-scale">
              <div className="w-12 h-12 rounded-lg gradient-bg flex items-center justify-center mb-4">
                <ArrowRight className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Export & Share</h3>
              <p className="text-muted-foreground">
                Save insights as notes, export to PDF, or share specific moments with your team or classmates.
              </p>
            </div>
          </div>
        </div>
      </section>
      
      {/* How It Works Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold">
              How <span className="gradient-text">VideoInsight</span> Works
            </h2>
            <p className="text-xl text-muted-foreground mt-4 max-w-3xl mx-auto">
              A simple three-step process to transform how you learn from video content.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            {/* Step 1 */}
            <div className="text-center">
              <div className="w-16 h-16 rounded-full gradient-bg flex items-center justify-center mx-auto mb-6">
                <span className="text-xl font-bold text-white">1</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Upload Your Videos</h3>
              <p className="text-muted-foreground">
                Drag and drop your educational videos into our secure platform. We support most video formats.
              </p>
            </div>
            
            {/* Step 2 */}
            <div className="text-center">
              <div className="w-16 h-16 rounded-full gradient-bg flex items-center justify-center mx-auto mb-6">
                <span className="text-xl font-bold text-white">2</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">AI Analysis</h3>
              <p className="text-muted-foreground">
                Our advanced AI processes the video content, transcribing speech and identifying key visual elements.
              </p>
            </div>
            
            {/* Step 3 */}
            <div className="text-center">
              <div className="w-16 h-16 rounded-full gradient-bg flex items-center justify-center mx-auto mb-6">
                <span className="text-xl font-bold text-white">3</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Interact & Extract</h3>
              <p className="text-muted-foreground">
                Ask questions, save insights, and navigate to important moments in your video content.
              </p>
            </div>
          </div>
        </div>
      </section>
      
      {/* CTA Section */}
      <section className="py-20 px-4 bg-gradient-to-br from-vidinsight-indigo via-vidinsight-purple to-vidinsight-blue">
        <div className="container mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
            Start Unlocking Video Insights Today
          </h2>
          <p className="text-xl text-white/80 mb-8 max-w-2xl mx-auto">
            Join the waitlist to be notified when the Playground is live!
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <form onSubmit={handleWaitlistSubmit} className="flex flex-col sm:flex-row gap-2 w-full sm:w-auto justify-center">
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="Enter your email"
                className={`px-4 py-2 rounded-l-md border border-white/30 focus:outline-none focus:ring-2 focus:ring-primary bg-background ${isDarkMode ? 'text-white placeholder-white/70' : 'text-black placeholder-black/70'}`}
                required
                disabled={submitted}
              />
              <Button
                type="submit"
                variant="secondary"
                className="rounded-r-md"
                disabled={submitted}
              >
                {submitted ? 'Joined!' : 'Join Waitlist'}
              </Button>
            </form>
          </div>
        </div>
      </section>
      
      {/* Waitlist Modal */}
      {showWaitlistModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-background rounded-xl shadow-2xl p-8 w-full max-w-md relative animate-fade-in">
            <button
              className="absolute top-4 right-4 text-2xl text-muted-foreground hover:text-foreground focus:outline-none"
              onClick={handleCloseWaitlistModal}
              aria-label="Close"
            >
              &times;
            </button>
            <h3 className="text-xl font-bold mb-4 text-center bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
              Enter your email to join the Waitlist
            </h3>
            <form onSubmit={handleWaitlistSubmit} className="flex flex-col gap-4">
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="Enter your email"
                className={`px-4 py-2 rounded-md border border-white/30 focus:outline-none focus:ring-2 focus:ring-primary bg-background ${isDarkMode ? 'text-white placeholder-white/70' : 'text-black placeholder-black/70'}`}
                required
                disabled={submitted}
                autoFocus
              />
              <Button
                type="submit"
                variant="secondary"
                className="rounded-md"
                disabled={submitted}
              >
                {submitted ? 'Joined!' : 'Join Waitlist'}
              </Button>
              {error && <div className="text-red-200 text-sm text-center">{error}</div>}
              {submitted && <div className="text-green-300 text-sm text-center">Thank you for joining! You'll be notified when the Playground is live.</div>}
            </form>
          </div>
        </div>
      )}
      
      <Footer />
    </div>
  );
};

export default Index;
