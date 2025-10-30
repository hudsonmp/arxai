const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

const supabase = createClient(
  'https://cbwylfksojmasehzmpum.supabase.co',
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNid3lsZmtzb2ptYXNlaHptcHVtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE1MDcxNjgsImV4cCI6MjA3NzA4MzE2OH0.PeuoW3JVqChd5pTBBbyhLg1eHcdSr93rGPNydLOrcVU'
);

const fallbackDir = path.join(__dirname, '..', 'fallback');

async function uploadPDFs() {
  console.log('\n📤 Uploading PDFs to Supabase...\n');
  
  const files = fs.readdirSync(fallbackDir);
  const annotatedPDFs = files.filter(f => f.endsWith('_annotated.pdf'));
  const originalPDFs = files.filter(f => f.endsWith('_original.pdf'));
  
  // Upload annotated PDFs
  console.log(`Annotated PDFs (${annotatedPDFs.length}):`);
  for (const filename of annotatedPDFs) {
    try {
      const filePath = path.join(fallbackDir, filename);
      const fileBuffer = fs.readFileSync(filePath);
      
      const { error } = await supabase.storage
        .from('arxai-annotated')
        .upload(filename, fileBuffer, {
          contentType: 'application/pdf',
          upsert: true
        });
      
      if (error) throw error;
      console.log(`   ✅ ${filename}`);
    } catch (error) {
      console.log(`   ❌ ${filename}: ${error.message}`);
    }
  }
  
  // Upload original PDFs
  console.log(`\nOriginal PDFs (${originalPDFs.length}):`);
  for (const filename of originalPDFs) {
    try {
      const filePath = path.join(fallbackDir, filename);
      const fileBuffer = fs.readFileSync(filePath);
      
      const { error } = await supabase.storage
        .from('arxai-originals')
        .upload(filename, fileBuffer, {
          contentType: 'application/pdf',
          upsert: true
        });
      
      if (error) throw error;
      console.log(`   ✅ ${filename}`);
    } catch (error) {
      console.log(`   ❌ ${filename}: ${error.message}`);
    }
  }
  
  console.log('\n✅ Upload complete!\n');
}

uploadPDFs().catch(console.error);

