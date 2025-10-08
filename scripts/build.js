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
  const slidesTemplatePath = path.join(__dirname, '../slides/templates/slides.html');
  const slidesTemplate = await fs.readFile(slidesTemplatePath, 'utf-8');
  
  const slides = await combineSlides();
  const distDir = path.join(__dirname, '../dist');
  await fs.mkdir(distDir, { recursive: true });
  
  const presentationFiles = [];
  
  // Check if there's only one slide file
  if (slides.length === 1) {
    // Generate single index.html for the presentation
    const slide = slides[0];
    const slidesContent = buildSlidesContent([slide]);
    const title = extractTitle(slide.content, slide.isHtml);
    
    let html = renderTemplate(slidesTemplate, {
      SLIDES_CONTENT: slidesContent,
      PRESENTATION_TITLE: title,
      HOT_RELOAD_SCRIPT: isDev ? '<script src="js/hot_reload.js"></script>' : ''
    });
    
    await fs.writeFile(path.join(distDir, 'index.html'), html);
    console.log(`✅ Generated index.html (single presentation)`);
  } else {
    // Generate a separate HTML file for each slide
    for (const slide of slides) {
      const slidesContent = buildSlidesContent([slide]);
      const title = extractTitle(slide.content, slide.isHtml);
      
      let html = renderTemplate(slidesTemplate, {
        SLIDES_CONTENT: slidesContent,
        PRESENTATION_TITLE: title,
        HOT_RELOAD_SCRIPT: isDev ? '<script src="js/hot_reload.js"></script>' : ''
      });
      
      // Get output filename (replace .md or .html extension with .html)
      const outputFilename = slide.filename.replace(/\.(md|html)$/, '.html');
      
      await fs.writeFile(path.join(distDir, outputFilename), html);
      console.log(`✅ Generated ${outputFilename}`);
      
      presentationFiles.push({
        filename: outputFilename,
        title: title
      });
    }
    
    // Generate index.html landing page
    await generateIndexPage(presentationFiles, distDir, isDev);
  }
  
  console.log('✅ All HTML files generated successfully!');
}

function extractTitle(content, isHtml) {
  let titleMatch = content.match(/<h1[^>]*>(.*?)<\/h1>/i);
  if (!titleMatch) {
    titleMatch = content.match(/^#\s+(.+)$/m);
  }
  const formatTitle = titleMatch ? titleMatch[1].replace(/<br\s*\/?>/gi, ' ').replace(/<[^>]*>/g, '').trim() : null;

  return formatTitle || 'Untitled Presentation';
}

async function generateIndexPage(presentationFiles, distDir, isDev) {
  const indexTemplatePath = path.join(__dirname, '../slides/templates/index.html');
  const indexTemplate = await fs.readFile(indexTemplatePath, 'utf-8');
  
  const presentationsList = presentationFiles.map(file => 
    `    <a href="${file.filename}" class="presentation-card">
      <h2>${file.title}</h2>
      <p class="filename">${file.filename}</p>
    </a>`
  ).join('\n');
  
  const indexHTML = renderTemplate(indexTemplate, {
    PRESENTATIONS_LIST: presentationsList,
    HOT_RELOAD_SCRIPT: isDev ? '<script src="js/hot_reload.js"></script>' : ''
  });

  await fs.writeFile(path.join(distDir, 'index.html'), indexHTML);
  console.log('✅ Generated index.html (landing page)');
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
  
  // Replace all placeholders
  for (const [key, value] of Object.entries(data)) {
    const placeholder = `{{${key}}}`;
    result = result.replace(new RegExp(placeholder, 'g'), value);
  }
  
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