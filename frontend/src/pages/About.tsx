import React from 'react';
import { Navbar } from '@/components/layout/Navbar';
import { useTheme } from '@/contexts/ThemeContext';
import { motion, useScroll, useTransform } from 'framer-motion';
import { Github, Linkedin, Twitter } from 'lucide-react';

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
              <div className="absolute inset-0 rounded-full gradient-bg opacity-20 animate-pulse"></div>
              <img
                src="/your-photo.jpg"
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
                href="https://github.com/yourusername"
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
                href="https://linkedin.com/in/yourusername"
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
                href="https://twitter.com/yourusername"
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
              href="mailto:your.email@example.com" 
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
              className="text-3xl font-bold mb-8"
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              My Story
            </motion.h2>
            <motion.div 
              className="prose prose-lg dark:prose-invert"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <p className="mb-6">
                As a passionate developer and AI enthusiast, I've always been fascinated by the intersection of technology and human learning. My journey in tech began with a simple curiosity about how machines can understand and process information like humans do.
              </p>
              <p className="mb-6">
                After working on various projects in web development and machine learning, I noticed a significant gap in how we interact with video content. While text-based information is easily searchable and analyzable, video content remains largely untapped in terms of its potential for knowledge extraction.
              </p>
              <p className="mb-6">
                This realization led me to create VideoInsight - a platform that bridges the gap between video content and actionable insights. My goal is to make video content as accessible and searchable as text, enabling people to learn more effectively from visual media.
              </p>
            </motion.div>
          </motion.section>

          {/* Mission Section */}
          <motion.section 
            className="mb-16"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
          >
            <motion.h2 
              className="text-3xl font-bold mb-8"
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              My Mission
            </motion.h2>
            <motion.div 
              className="prose prose-lg dark:prose-invert"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <p className="mb-6">
                I believe that knowledge should be accessible to everyone, and video content is one of the most powerful mediums for learning. However, the current state of video content makes it difficult to extract specific information or find relevant sections quickly.
              </p>
              <p className="mb-6">
                With VideoInsight, I'm working to revolutionize how we interact with video content by:
              </p>
              <motion.ul 
                className="list-disc pl-6 mb-6 space-y-2"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.6 }}
              >
                <motion.li 
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: 0.7 }}
                >
                  Making video content searchable and analyzable
                </motion.li>
                <motion.li 
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: 0.8 }}
                >
                  Enabling quick access to specific information within videos
                </motion.li>
                <motion.li 
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: 0.9 }}
                >
                  Providing insights and summaries of video content
                </motion.li>
                <motion.li 
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: 1 }}
                >
                  Creating a more efficient learning experience
                </motion.li>
              </motion.ul>
              <motion.p 
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 1.1 }}
              >
                I'm committed to building an open-source community around this project, where developers and researchers can collaborate to improve video content accessibility for everyone.
              </motion.p>
            </motion.div>
          </motion.section>
        </div>
      </main>
    </div>
  );
} 