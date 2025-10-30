'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';

interface PaperPair {
  name: string;
  originalUrl: string;
  annotatedUrl: string;
}

export default function Home() {
  const [papers, setPapers] = useState<PaperPair[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showAnnotated, setShowAnnotated] = useState(true);

  useEffect(() => {
    loadPapers();
  }, []);

  const loadPapers = async () => {
    try {
      const { data: annotated } = await supabase.storage
        .from('arxai-annotated')
        .list('', { limit: 100, sortBy: { column: 'created_at', order: 'desc' } });

      if (annotated) {
        const paperPairs = annotated
          .filter(file => file.name.endsWith('_annotated.pdf'))
          .map(file => {
            const baseName = file.name.replace('_annotated.pdf', '');
            return {
              name: baseName,
              originalUrl: supabase.storage.from('arxai-originals').getPublicUrl(`${baseName}_original.pdf`).data.publicUrl,
              annotatedUrl: supabase.storage.from('arxai-annotated').getPublicUrl(file.name).data.publicUrl,
            };
          });
        setPapers(paperPairs);
      }
    } catch (error) {
      console.error('Error loading papers:', error);
    } finally {
      setLoading(false);
    }
  };

  const goToNext = () => {
    if (currentIndex < papers.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const goToPrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const currentPaper = papers[currentIndex];

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-slate-200 border-t-slate-900 mb-6"></div>
          <p className="text-slate-700 text-lg font-medium">Loading papers...</p>
        </div>
      </div>
    );
  }

  if (papers.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <div className="text-8xl mb-6 opacity-20">📄</div>
          <p className="text-slate-700 text-lg font-medium">No papers found</p>
          <p className="text-slate-500 text-sm mt-2">Upload some PDFs to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="h-screen flex flex-col max-w-7xl mx-auto">
        {/* Header */}
        <header className="px-8 py-6 bg-white/80 backdrop-blur-sm border-b border-slate-200">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">
                📄 Paper Annotation Viewer
              </h1>
              <p className="text-sm text-slate-600 mt-1">
                {currentPaper.name.replace(/-/g, ' ')}
              </p>
            </div>
            <div className="text-right">
              <div className="text-sm font-medium text-slate-700">
                Paper {currentIndex + 1} of {papers.length}
              </div>
            </div>
          </div>
        </header>

        {/* Controls Bar */}
        <div className="px-8 py-4 bg-white/60 backdrop-blur-sm border-b border-slate-200">
          <div className="flex items-center justify-between gap-4">
            {/* Navigation */}
            <div className="flex gap-2">
              <button
                onClick={goToPrevious}
                disabled={currentIndex === 0}
                className="px-4 py-2 bg-slate-900 text-white rounded-lg hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition-all text-sm font-medium flex items-center gap-2"
              >
                ← Previous
              </button>
              <button
                onClick={goToNext}
                disabled={currentIndex === papers.length - 1}
                className="px-4 py-2 bg-slate-900 text-white rounded-lg hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition-all text-sm font-medium flex items-center gap-2"
              >
                Next →
              </button>
            </div>

            {/* Toggle View */}
            <div className="flex items-center gap-3 bg-slate-100 rounded-lg p-1">
              <button
                onClick={() => setShowAnnotated(false)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  !showAnnotated 
                    ? 'bg-white text-slate-900 shadow-sm' 
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Original
              </button>
              <button
                onClick={() => setShowAnnotated(true)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  showAnnotated 
                    ? 'bg-white text-slate-900 shadow-sm' 
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                ✨ Annotated
              </button>
            </div>

            {/* Download */}
            <div className="flex gap-2">
              <a
                href={showAnnotated ? currentPaper.annotatedUrl : currentPaper.originalUrl}
                download
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all text-sm font-medium"
              >
                Download PDF
              </a>
            </div>
          </div>
        </div>

        {/* PDF Viewer */}
        <div className="flex-1 px-8 py-6 overflow-hidden">
          <div className="h-full bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden">
            <iframe
              src={showAnnotated ? currentPaper.annotatedUrl : currentPaper.originalUrl}
              className="w-full h-full border-0"
              title={showAnnotated ? 'Annotated PDF' : 'Original PDF'}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
