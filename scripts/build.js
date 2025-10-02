#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');

async function combineSlides() {
  const slidesDir = path.join(__dirname, '../slides');
  const slideFiles = await fs.readdir(slidesDir);
  
  const sortedFiles = slideFiles
    .filter(file => file.endsWith('.md') || (file.endsWith('.html') && file !== 'index.html'))
    .sort();
  
  let slides = [];
  
  for (const file of sortedFiles) {
    const filePath = path.join(slidesDir, file);
    const content = await fs.readFile(filePath, 'utf-8');
    
    slides.push({
      filename: file,
      content: content,
      isHtml: file.endsWith('.html')
    });
  }
  
  return slides;
}

async function generateHTML(isDev = false) {
  const templatePath = path.join(__dirname, '../slides/templates/index.html');
  const template = await fs.readFile(templatePath, 'utf-8');
  
  const slides = await combineSlides();
  const slidesContent = buildSlidesContent(slides);

  // Use a simple template function instead of string replace
  const html = renderTemplate(template, {
    slidesContent: slidesContent,
    isDev: isDev,
  });
  
  const distDir = path.join(__dirname, '../dist');
  await fs.mkdir(distDir, { recursive: true });
  
  await fs.writeFile(path.join(distDir, 'index.html'), html);
  
  console.log('✅ HTML generated successfully!');
}

function buildSlidesContent(slides) {
  let slidesContent = '';
  let markdownContent = '';
  let isInMarkdownSection = false;
  
  for (let i = 0; i < slides.length; i++) {
    const slide = slides[i];
    
    if (slide.isHtml) {
      // Close markdown section if we're in one
      if (isInMarkdownSection) {
        slidesContent += createMarkdownSection(markdownContent);
        markdownContent = '';
        isInMarkdownSection = false;
      }
      // Add HTML slide
      slidesContent += slide.content + '\n      ';
    } else {
      // Add markdown content
      markdownContent += slide.content + '\n\n---\n\n';
      isInMarkdownSection = true;
    }
  }
  
  // Close final markdown section if needed
  if (isInMarkdownSection) {
    slidesContent += createMarkdownSection(markdownContent);
  }
  
  return slidesContent;
}

function createMarkdownSection(markdownContent) {
  const cleanContent = markdownContent.replace(/\n\n---\n\n$/, '');
  return `<section data-markdown data-separator="^---" data-separator-vertical="^--">
        <textarea data-template>
${cleanContent}
        </textarea>
      </section>
      `;
}

function renderTemplate(template, data) {
  let result = template;
  
  // Replace SLIDES_CONTENT
  if (data.slidesContent) {
    result = result.replace(/\{\{SLIDES_CONTENT\}\}/g, data.slidesContent);
  }

  let hotReloadScript = data.isDev ? '<script src="js/hot_reload.js"></script>' : '';

  // Replace HOT_RELOAD_SCRIPT
  result = result.replace(/\{\{HOT_RELOAD_SCRIPT\}\}/g, hotReloadScript);
  
  return result;
}

async function copyAllFolders() {
  const jsDir = path.join(__dirname, '../slides/js');
  const cssDir = path.join(__dirname, '../slides/css');
  const assetsDir = path.join(__dirname, '../slides/assets');
  const distDir = path.join(__dirname, '../dist');
  
  // Create dist directory
  await fs.mkdir(distDir, { recursive: true });
  
  // Copy all folders from slides/src to dist
  const entries = { jsDir, cssDir, assetsDir };
  for (const [key, srcDir] of Object.entries(entries)) {
    const destDir = path.join(distDir, key.replace('Dir', ''));
    try {
      await copyDirectory(srcDir, destDir);
      console.log(`📂 Copied ${key.replace('Dir', '')}/`);
    } catch (error) {
      console.error(`❌ Failed to copy ${key.replace('Dir', '')}:`, error.message);
    }
  }
  
  // Copy reveal.js from node_modules
  const nodeModulesReveal = path.join(__dirname, '../node_modules/reveal.js');
  const distReveal = path.join(distDir, 'reveal.js');
  
  try {
    // Copy reveal.js core files
    const revealDistDir = path.join(nodeModulesReveal, 'dist');
    const targetDistDir = path.join(distReveal, 'dist');
    await copyDirectory(revealDistDir, targetDistDir);
    
    // Copy plugins
    const pluginDir = path.join(nodeModulesReveal, 'plugin');
    const targetPluginDir = path.join(distReveal, 'plugin');
    await copyDirectory(pluginDir, targetPluginDir);
    
    console.log('📂 Copied reveal.js/');
  } catch (error) {
    console.error('❌ Failed to copy reveal.js files:', error.message);
  }
  
  // Copy KaTeX from node_modules
  const nodeModulesKatex = path.join(__dirname, '../node_modules/katex');
  const distKatex = path.join(distDir, 'katex');
  
  try {
    // Copy KaTeX dist files
    const katexDistDir = path.join(nodeModulesKatex, 'dist');
    const targetDistDir = path.join(distKatex, 'dist');
    await copyDirectory(katexDistDir, targetDistDir);

    console.log('📂 Copied katex/');
  } catch (error) {
    console.error('❌ Failed to copy katex files:', error.message);
  }
}

async function copyDirectory(src, dest) {
  await fs.mkdir(dest, { recursive: true });
  const entries = await fs.readdir(src, { withFileTypes: true });
  
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    
    if (entry.isDirectory()) {
      await copyDirectory(srcPath, destPath);
    } else {
      await fs.copyFile(srcPath, destPath);
    }
  }
}

async function build(isDev = false) {
  try {
    console.log('🔨 Building presentation...');
    await generateHTML(isDev);
    await copyAllFolders();
    console.log('✅ Build process completed!');
  } catch (error) {
    console.error('❌ Build failed:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  build();
}

module.exports = { build, combineSlides };