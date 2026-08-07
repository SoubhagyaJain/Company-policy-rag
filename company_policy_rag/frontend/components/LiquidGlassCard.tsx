'use client';

import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { cn } from '../lib/utils';

interface LiquidGlassCardProps extends HTMLMotionProps<'div'> {
  children: React.ReactNode;
  className?: string;
  variant?: 'cream' | 'sand' | 'dark' | 'glow';
  hoverEffect?: boolean;
}

export function LiquidGlassCard({
  children,
  className,
  variant = 'cream',
  hoverEffect = false,
  ...props
}: LiquidGlassCardProps) {
  const getVariantStyles = () => {
    switch (variant) {
      case 'sand':
        return 'bg-sand-light/90 dark:bg-sand-dark/90 border-sand-border dark:border-sand-darkBorder';
      case 'dark':
        return 'bg-charcoal/90 dark:bg-charcoal-dark/90 border-charcoal-light/20 text-cream-100';
      case 'glow':
        return 'bg-cream-50/90 dark:bg-cream-950/90 border-terracotta-500/30 shadow-[0_0_25px_rgba(217,119,6,0.1)]';
      case 'cream':
      default:
        return 'bg-[#FAF9F5]/85 dark:bg-[#141413]/85 border-[#E5E0D8]/70 dark:border-[#2A2925]/70';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      whileHover={hoverEffect ? { y: -2, transition: { duration: 0.15 } } : undefined}
      className={cn(
        'backdrop-blur-md rounded-2xl border shadow-[0_4px_20px_-2px_rgba(26,26,26,0.04)] dark:shadow-[0_4px_25px_-2px_rgba(0,0,0,0.3)] transition-all',
        getVariantStyles(),
        className
      )}
      {...props}
    >
      {children}
    </motion.div>
  );
}
