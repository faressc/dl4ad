#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');
const { marked } = require('marked');

async function combineSlides() {
  const slidesDir = path.join(__dirname, '../slides/src/content');
  const slideFiles = await fs.readdir(slidesDir);
  
  const sortedFiles = slideFiles
    .filter(file => file.endsWith('.md') || file.endsWith('.html'))
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
  
  // Build the slides content with proper ordering
  let slidesContent = '';
  let markdownContent = '';
  let isInMarkdownSection = false;
  
  for (let i = 0; i < slides.length; i++) {
    const slide = slides[i];
    
    if (slide.isHtml) {
      // Close markdown section if we're in one
      if (isInMarkdownSection) {
        slidesContent += '<section data-markdown data-separator="^---" data-separator-vertical="^--">\n        <textarea data-template>\n' + markdownContent.replace(/\n\n---\n\n$/, '') + '\n        </textarea>\n      </section>\n      ';
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
    slidesContent += '<section data-markdown data-separator="^---" data-separator-vertical="^--">\n        <textarea data-template>\n' + markdownContent.replace(/\n\n---\n\n$/, '') + '\n        </textarea>\n      </section>';
  }
  
  // Add hot reload script only in development
  const hotReloadScript = isDev ? `
    // Hot reload WebSocket connection
    const ws = new WebSocket(\`ws://\${window.location.host}\`);
    ws.onmessage = function(event) {
      if (event.data === 'reload') {
        window.location.reload();
      }
    };
    ws.onopen = function() {
      console.log('🔥 Hot reload connected');
    };
    ws.onclose = function() {
      console.log('🔌 Hot reload disconnected');
    };` : '';

  const html = template
    .replace('{{SLIDES_CONTENT}}', slidesContent)
    .replace('{{HOT_RELOAD_SCRIPT}}', hotReloadScript);
  
  const distDir = path.join(__dirname, '../dist');
  await fs.mkdir(distDir, { recursive: true });
  
  await fs.writeFile(path.join(distDir, 'index.html'), html);
  
  const srcThemePath = path.join(__dirname, '../slides/src/themes/custom-theme.css');
  const distThemePath = path.join(distDir, 'custom-theme.css');
  
  const themeCSS = await fs.readFile(srcThemePath, 'utf-8');
  await fs.writeFile(distThemePath, themeCSS);
  
  // Copy reveal-config.js
  const configSrcPath = path.join(__dirname, '../slides/src/config/reveal-config.js');
  const configDistPath = path.join(distDir, 'reveal-config.js');
  const configContent = await fs.readFile(configSrcPath, 'utf-8');
  await fs.writeFile(configDistPath, configContent);
  
  console.log('✅ Build completed successfully!');
  console.log('📁 Output: dist/index.html');
}

async function copyAssets() {
  const assetsDir = path.join(__dirname, '../slides/src/assets');
  const distAssetsDir = path.join(__dirname, '../dist/assets');
  
  try {
    await fs.mkdir(distAssetsDir, { recursive: true });
    
    const assetTypes = ['images', 'videos', 'fonts'];
    
    for (const assetType of assetTypes) {
      const srcPath = path.join(assetsDir, assetType);
      const destPath = path.join(distAssetsDir, assetType);
      
      try {
        const files = await fs.readdir(srcPath);
        if (files.length > 0) {
          await fs.mkdir(destPath, { recursive: true });
          
          for (const file of files) {
            const srcFile = path.join(srcPath, file);
            const destFile = path.join(destPath, file);
            await fs.copyFile(srcFile, destFile);
          }
          console.log(`📂 Copied ${assetType}`);
        }
      } catch (error) {
        // Directory doesn't exist or is empty, skip
      }
    }
  } catch (error) {
    console.log('ℹ️  No assets to copy');
  }
}

async function copyRevealJS() {
  const nodeModulesReveal = path.join(__dirname, '../node_modules/reveal.js');
  const distReveal = path.join(__dirname, '../dist/reveal.js');
  
  try {
    await fs.mkdir(distReveal, { recursive: true });
    
    // Copy reveal.js core files
    const distDir = path.join(nodeModulesReveal, 'dist');
    const targetDistDir = path.join(distReveal, 'dist');
    await copyDirectory(distDir, targetDistDir);
    
    // Copy plugins
    const pluginDir = path.join(nodeModulesReveal, 'plugin');
    const targetPluginDir = path.join(distReveal, 'plugin');
    await copyDirectory(pluginDir, targetPluginDir);
    
    console.log('📂 Copied reveal.js files');
  } catch (error) {
    console.error('❌ Failed to copy reveal.js files:', error.message);
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
    await copyRevealJS();
    await copyAssets();
    console.log('🎉 Build process completed!');
  } catch (error) {
    console.error('❌ Build failed:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  build();
}

module.exports = { build, combineSlides };