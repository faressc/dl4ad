// Inject custom CSS into speaker notes window
const originalOpen = window.open;
window.open = function(url, name, specs) {
    const notesWindow = originalOpen.call(this, url, name, specs);
    
    // Inject CSS for speaker notes (opens as about:blank initially)
    if (name === 'reveal.js - Notes') {
      console.log('Injecting custom CSS into notes window');
      
      // Multiple strategies to detect when notes window is ready
      const waitForNotesWindow = async () => {
        const maxWaitTime = 10000; // 10 seconds max wait
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWaitTime) {
          try {
            if (notesWindow && notesWindow.document && notesWindow.document.head && notesWindow.document.body) {
              // Check window location (it should change from about:blank), then the window is ready to inject CSS
              const hasUrl = notesWindow.location && notesWindow.location.href !== 'about:blank';

              if (hasUrl) {
                console.log('Notes window appears ready, injecting CSS');
                
                // Inject the CSS
                const existingStyle = notesWindow.document.querySelector('link[href*="speaker_notes"]');
                if (!existingStyle) {
                  const style = notesWindow.document.createElement('link');
                  style.rel = 'stylesheet';
                  style.href = 'css/speaker_notes.css';
                  style.type = 'text/css';
                  notesWindow.document.head.appendChild(style);
               
                  console.log('CSS injection completed successfully');
                  return true;
                }
              }
            }
          } catch (error) {
            console.log('Error checking window state:', error);
          }

          // Wait 100ms before checking again
          await new Promise(resolve => setTimeout(resolve, 100));
          console.log('Waiting for notes window to load...');
        }
        
        console.log('Timeout waiting for notes window to load');
        return false;
      };
      
      waitForNotesWindow();
    }
    
    return notesWindow;
};