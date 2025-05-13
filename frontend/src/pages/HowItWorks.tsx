import React from 'react';
import { useTheme } from '@/contexts/ThemeContext';
import { Navbar } from '@/components/layout/Navbar';
import { motion } from 'framer-motion';
import { 
  Upload, 
  Split, 
  Search, 
  MessageSquare, 
  Video, 
  Sparkles,
  ArrowRight,
  Code2,
  Database,
  Brain,
  Network,
  Layers
} from 'lucide-react';

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 }
};

const HowItWorks = () => {
  const { toggleTheme, isDarkMode } = useTheme();

  const steps = [
    {
      icon: Upload,
      title: "Upload Your Video",
      description: "Start by uploading any video file or providing a YouTube URL. Our system supports various video formats and handles both short clips and longer content.",
      color: "from-vidinsight-indigo to-vidinsight-purple"
    },
    {
      icon: Split,
      title: "Intelligent Processing",
      description: "The video is automatically split into meaningful segments, and key frames are extracted. Our AI analyzes the content to create a comprehensive understanding of your video.",
      color: "from-vidinsight-purple to-vidinsight-blue"
    },
    {
      icon: Search,
      title: "Advanced Indexing",
      description: "Using state-of-the-art embedding models, we create a searchable index of your video content. This enables lightning-fast retrieval of relevant segments.",
      color: "from-vidinsight-blue to-vidinsight-indigo"
    },
    {
      icon: MessageSquare,
      title: "Natural Language Interaction",
      description: "Ask questions about your video in natural language. Our system understands context and provides precise answers with relevant video segments.",
      color: "from-vidinsight-indigo to-vidinsight-purple"
    },
    {
      icon: Video,
      title: "Smart Clip Generation",
      description: "Generate custom clips based on your queries. The system identifies the most relevant segments and creates concise, focused video clips.",
      color: "from-vidinsight-purple to-vidinsight-blue"
    }
  ];

  const techStack = [
    {
      category: "Frontend",
      technologies: [
        { name: "Next.js", description: "React framework for server-side rendering and static site generation" },
        { name: "TypeScript", description: "Type-safe JavaScript for robust development" },
        { name: "Tailwind CSS", description: "Utility-first CSS framework for rapid UI development" },
        { name: "Shadcn UI", description: "Re-usable components built with Radix UI and Tailwind" }
      ]
    },
    {
      category: "Backend",
      technologies: [
        { name: "Python", description: "Core processing and AI pipeline implementation" },
        { name: "FastAPI", description: "High-performance API framework for Python" },
        { name: "LanceDB", description: "Vector database for efficient similarity search" },
        { name: "FFmpeg", description: "Video processing and manipulation" }
      ]
    },
    {
      category: "AI/ML",
      technologies: [
        { name: "LLaVA", description: "Large Language and Vision Assistant for multimodal understanding" },
        { name: "Sentence Transformers", description: "Text embedding models for semantic search" },
        { name: "Pytorch", description: "Deep learning framework for model inference" },
        { name: "Hugging Face", description: "Model hub and inference pipeline" }
      ]
    }
  ];

  const architectureComponents = [
    {
      icon: Layers,
      title: "Video Processing Pipeline",
      description: "Our system processes videos through a basic pipeline:",
      details: [
        "Frame extraction at 1-second intervals using OpenCV",
        "Support for YouTube videos and local uploads",
        "Basic video splitting capabilities",
        "Temporary storage management for processed videos"
      ]
    },
    {
      icon: Brain,
      title: "Multimodal Understanding",
      description: "The system uses a combination of models for understanding:",
      details: [
        "BridgeTower for generating multimodal embeddings",
        "GPT-4V integration for visual understanding",
        "Basic RAG implementation for context-aware responses",
        "Frame-text pair processing for unified understanding"
      ]
    },
    {
      icon: Database,
      title: "Vector Database Architecture",
      description: "We use LanceDB for vector storage:",
      details: [
        "Basic vector storage for frame embeddings",
        "Simple similarity search implementation",
        "Metadata storage for frame timestamps",
        "Temporary storage for processed videos"
      ]
    },
    {
      icon: Network,
      title: "API Architecture",
      description: "The system provides basic API functionality:",
      details: [
        "Video upload and processing endpoints",
        "Query processing for video analysis",
        "Basic error handling and validation",
        "Session management for video processing"
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navbar toggleTheme={toggleTheme} isDarkMode={isDarkMode} />

      {/* Hero Section */}
      <motion.div 
        className="relative pt-32 pb-20 px-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="container mx-auto max-w-4xl text-center">
          <motion.h1 
            className="text-4xl md:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-vidinsight-indigo via-vidinsight-purple to-vidinsight-blue"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            How VideoInsight Works
          </motion.h1>
          <motion.p 
            className="text-xl text-muted-foreground mb-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            Transform your video content into an interactive, searchable knowledge base with our advanced AI technology.
          </motion.p>
        </div>
      </motion.div>

      {/* Process Steps */}
      <div className="container mx-auto max-w-5xl px-4 pb-20">
        <div className="space-y-20">
          {steps.map((step, index) => (
            <motion.div
              key={step.title}
              className="relative"
              initial="initial"
              whileInView="animate"
              viewport={{ once: true, margin: "-100px" }}
              variants={fadeInUp}
            >
              <div className="flex flex-col md:flex-row items-center gap-8">
                {/* Icon */}
                <div className={`w-20 h-20 rounded-2xl bg-gradient-to-br ${step.color} flex items-center justify-center flex-shrink-0`}>
                  <step.icon className="w-10 h-10 text-white" />
                </div>

                {/* Content */}
                <div className="flex-1 text-center md:text-left">
                  <h3 className="text-2xl font-bold mb-3">{step.title}</h3>
                  <p className="text-muted-foreground">{step.description}</p>
                </div>

                {/* Arrow (except for last step) */}
                {index < steps.length - 1 && (
                  <div className="hidden md:block absolute left-1/2 top-full -translate-x-1/2 mt-4">
                    <ArrowRight className="w-6 h-6 text-muted-foreground rotate-90" />
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Features Section */}
      <motion.div 
        className="bg-muted/30 py-20"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
      >
        <div className="container mx-auto max-w-4xl px-4">
          <motion.h2 
            className="text-3xl font-bold text-center mb-12"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            Key Features
          </motion.h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            {[
              {
                title: "Real-time Processing",
                description: "Upload and process videos instantly with our optimized pipeline."
              },
              {
                title: "Smart Segmentation",
                description: "Intelligent video splitting that maintains context and meaning."
              },
              {
                title: "Contextual Understanding",
                description: "Deep comprehension of video content for accurate responses."
              },
              {
                title: "Custom Clip Generation",
                description: "Create focused video clips based on your specific needs."
              }
            ].map((feature, index) => (
              <motion.div
                key={feature.title}
                className="bg-background p-6 rounded-lg shadow-sm"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <h3 className="font-semibold mb-2">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Technical Deep Dive Section */}
      <motion.div 
        className="py-20"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
      >
        <div className="container mx-auto max-w-6xl px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Technical Architecture</h2>
            <p className="text-muted-foreground text-lg max-w-3xl mx-auto">
              A deep dive into the technologies and architectural decisions that power VideoInsight
            </p>
          </motion.div>

          {/* Tech Stack */}
          <div className="grid md:grid-cols-3 gap-8 mb-20">
            {techStack.map((category, index) => (
              <motion.div
                key={category.category}
                className="bg-muted/30 rounded-lg p-6"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <h3 className="text-xl font-bold mb-4">{category.category}</h3>
                <ul className="space-y-4">
                  {category.technologies.map(tech => (
                    <li key={tech.name}>
                      <div className="font-semibold">{tech.name}</div>
                      <div className="text-sm text-muted-foreground">{tech.description}</div>
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>

          {/* Architecture Components */}
          <div className="space-y-12">
            {architectureComponents.map((component, index) => (
              <motion.div
                key={component.title}
                className="bg-background rounded-lg p-8 shadow-sm"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="flex items-start gap-6">
                  <div className="w-12 h-12 rounded-xl gradient-bg flex items-center justify-center flex-shrink-0">
                    <component.icon className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold mb-3">{component.title}</h3>
                    <p className="text-muted-foreground mb-4">{component.description}</p>
                    <ul className="space-y-2">
                      {component.details.map((detail, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                          <span>{detail}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* CTA Section */}
      <motion.div 
        className="py-20 text-center"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
      >
        <div className="container mx-auto max-w-2xl px-4">
          <motion.div
            className="inline-block"
            initial={{ scale: 0.9 }}
            whileInView={{ scale: 1 }}
            viewport={{ once: true }}
          >
            <div className="w-16 h-16 rounded-full gradient-bg flex items-center justify-center mx-auto mb-6">
              <Sparkles className="w-8 h-8 text-white" />
            </div>
          </motion.div>
          
          <motion.h2 
            className="text-3xl font-bold mb-4"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            Ready to Transform Your Video Content?
          </motion.h2>
          
          <motion.p 
            className="text-muted-foreground mb-8"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
            Start exploring the future of video interaction today.
          </motion.p>
        </div>
      </motion.div>
    </div>
  );
};

export default HowItWorks; 