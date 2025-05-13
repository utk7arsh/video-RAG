import React from 'react';
import { Navbar } from '@/components/layout/Navbar';
import { useTheme } from '@/contexts/ThemeContext';
import { motion, useScroll, useTransform } from 'framer-motion';
import { Github, Linkedin, Twitter } from 'lucide-react';
import meImg from '@/me.jpg';

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 }
};

const socialLinkVariants = {
  initial: { scale: 1 },
  hover: { 
    scale: 1.1,
    transition: { duration: 0.2 }
  }
};

export default function About() {
  const { isDarkMode, toggleTheme } = useTheme();
  const { scrollYProgress } = useScroll();

  return (
    <div className="min-h-screen bg-background">
      <Navbar toggleTheme={toggleTheme} isDarkMode={isDarkMode} />
      
      <main className="pt-20 flex">
        {/* Fixed Profile Section */}
        <div className="w-1/3 min-h-screen p-8 border-r border-border fixed top-20 left-0">
          <motion.div 
            className="max-w-sm mx-auto"
            initial="initial"
            animate="animate"
            variants={fadeInUp}
          >
            <motion.div 
              className="relative w-48 h-48 mx-auto mb-8"
              variants={fadeInUp}
            >
              <div className="absolute inset-0 rounded-full gradient-bg opacity-20"></div>
              <img
                src={meImg}
                alt="Utkarsh"
                className="w-full h-full object-cover rounded-full border-4 border-background"
              />
            </motion.div>
            <motion.h1 
              className="text-4xl font-bold mb-4 bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent text-center"
              variants={fadeInUp}
            >
              Hi, I'm Utkarsh
            </motion.h1>
            <motion.p 
              className="text-xl text-muted-foreground mb-8 text-center"
              variants={fadeInUp}
            >
              Full Stack Developer & AI Enthusiast
            </motion.p>
            <motion.div 
              className="flex justify-center space-x-6 mb-8"
              variants={fadeInUp}
            >
              <motion.a
                href="https://github.com/utk7arsh"
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 rounded-full border border-border hover:border-primary/50 transition-colors"
                variants={socialLinkVariants}
                whileHover="hover"
                whileTap={{ scale: 0.95 }}
              >
                <Github className="w-6 h-6 text-muted-foreground hover:text-foreground transition-colors" />
              </motion.a>
              <motion.a
                href="https://www.linkedin.com/in/utkarshlal/"
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 rounded-full border border-border hover:border-primary/50 transition-colors"
                variants={socialLinkVariants}
                whileHover="hover"
                whileTap={{ scale: 0.95 }}
              >
                <Linkedin className="w-6 h-6 text-muted-foreground hover:text-foreground transition-colors" />
              </motion.a>
              <motion.a
                href="https://x.com/utk7arsh"
                target="_blank"
                rel="noopener noreferrer"
                className="p-3 rounded-full border border-border hover:border-primary/50 transition-colors"
                variants={socialLinkVariants}
                whileHover="hover"
                whileTap={{ scale: 0.95 }}
              >
                <Twitter className="w-6 h-6 text-muted-foreground hover:text-foreground transition-colors" />
              </motion.a>
            </motion.div>
            <motion.a 
              href="mailto:utkarshlal@gmail.com" 
              className="block w-full text-center px-6 py-3 rounded-full gradient-bg text-white font-medium hover:opacity-90 transition-opacity"
              variants={fadeInUp}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Get in Touch
            </motion.a>
          </motion.div>
        </div>

        {/* Scrollable Content Section */}
        <div className="w-2/3 p-8 ml-[33.333333%]">
          {/* Story Section */}
          <motion.section 
            className="mb-16"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            <motion.h2 
              className="text-4xl md:text-5xl font-extrabold mb-8 bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent drop-shadow-sm tracking-tight"
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              My Story
            </motion.h2>
            <motion.div 
              className="prose prose-lg dark:prose-invert max-w-2xl text-lg leading-relaxed"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <p>
                Hey, I'm <span className="font-bold text-primary">Utkarsh</span> — a builder at heart and a relentless tinkerer at the intersection of <span className="font-semibold text-gradient bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">AI, systems, and learning</span>. I am a senior at <span className="font-semibold text-blue-500">University of California, Los Angeles</span>. My journey into tech wasn't sparked by a single <span className="italic">"aha"</span> moment, but by a growing obsession with how humans absorb information, and how badly our current tools bottleneck that process.
              </p>
              <p>
                As someone who's spent years working across <span className="font-semibold text-gradient bg-gradient-to-r from-purple-400 to-blue-500 bg-clip-text text-transparent">machine learning, full-stack development, and research</span>, I kept running into the same problem: <span className="font-bold text-primary">video content is still a black box</span>. While we've nailed search and analysis for text, video remains clunky, hard to navigate, and nearly impossible to query meaningfully.
              </p>
              <p>
                That's why I created <span className="font-bold text-gradient bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">VideoInsight</span> — a system designed to make video content as interactive and explorable as a conversation. Imagine being able to <span className="font-semibold text-primary">ask a video a question and instantly jump to the answer</span>, not just get a vague summary. That's what I'm building.
              </p>

              <p>
                Whether you're a student, creator, researcher, or just curious — I want you to spend less time scrubbing timelines and more time learning. <span className="font-semibold text-gradient bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">Please reach out to me if you'd like to see any new features in the app. Make sure to join the waitlist to be notified when the Playground is live!</span>
              </p>
            </motion.div>
          </motion.section>
        </div>
      </main>
    </div>
  );
} 