/**
 * Timeline.js - Interactive Timeline Testing Script
 * Provides functionality to test and manipulate SVG lines in timeline presentations
 */

console.log('Timeline.js loading...');

// Timeline Controller Class
class TimelineController {
    constructor() {
        this.lines = [];
        this.isReady = false;
        this.animations = new Map();
        
        // Wait for Reveal.js to be ready
        window.addEventListener('reveal-ready', () => {
            this.initialize();
        });
    }
    
    initialize() {
        console.log('Timeline Controller initializing...');
        this.findTimelineElements();
        this.processLines();
        this.isReady = true;
        console.log('Timeline Controller ready!');
    }
    
    findTimelineElements() {
        // Find all SVG lines in timeline containers
        const timelineLines = document.querySelectorAll('.timeline-connector line');
        
        this.lines = Array.from(timelineLines).map((line, index) => ({
            element: line,
            id: index,
            originalX1: line.getAttribute('x1'),
            originalY1: line.getAttribute('y1'),
            originalX2: line.getAttribute('x2'),
            originalY2: line.getAttribute('y2'),
            year: this.getCSSVariable(line, '--year') || null,
            startYear: this.getCSSVariable(line, '--start-year') || null,
            endYear: this.getCSSVariable(line, '--end-year') || null,
            itemIndex: this.getCSSVariable(line, '--item-index') || null,
            itemsInTimeline: this.countItemsInTimeline(line)
        }));
        
        console.log(`Found ${this.lines.length} timeline lines:`, this.lines);
    }
    
    processLines() {
        for (const lineData of this.lines) {
            const { element, id, year, startYear, endYear, itemIndex, itemsInTimeline } = lineData;
            console.log(`Processing line ID ${id}: year=${year}, startYear=${startYear}, endYear=${endYear}`);
            const x1 = (year - startYear) / (endYear - startYear) * (100 - 2.2) + 1.1;
            this.setLineX1(id, `${x1}%`);
            let x2 = (parseFloat(itemIndex) + 0.5) / itemsInTimeline * 100;
            this.setLineX2(id, `${x2}%`);
            this.setLineY1(id, "20px");
            this.setLineY2(id, "35px");
        }
    }

    // Core line manipulation methods
    setLineCoordinate(lineId, coordinate, value) {
        if (!this.isValidLine(lineId)) return false;
        
        const line = this.lines[lineId].element;
        line.setAttribute(coordinate, value);
        return true;
    }
    
    getLineCoordinate(lineId, coordinate) {
        if (!this.isValidLine(lineId)) return null;
        
        const line = this.lines[lineId].element;
        return line.getAttribute(coordinate);
    }
    
    setLineX1(lineId, x1) {
        return this.setLineCoordinate(lineId, 'x1', x1);
    }
    
    setLineY1(lineId, y1) {
        return this.setLineCoordinate(lineId, 'y1', y1);
    }
    
    setLineX2(lineId, x2) {
        return this.setLineCoordinate(lineId, 'x2', x2);
    }
    
    setLineY2(lineId, y2) {
        return this.setLineCoordinate(lineId, 'y2', y2);
    }

    // CSS Variable helper methods
    getCSSVariable(element, variableName) {
        // Method 1: Get computed style value
        const computedStyle = getComputedStyle(element);
        let value = computedStyle.getPropertyValue(variableName);
        
        if (value) {
            return value.trim();
        }
        
        // Method 2: Check parent elements up the tree
        let parent = element.parentElement;
        while (parent) {
            const parentStyle = getComputedStyle(parent);
            value = parentStyle.getPropertyValue(variableName);
            if (value) {
                return value.trim();
            }
            parent = parent.parentElement;
        }
        
        // Method 3: Check inline style (if set via style attribute)
        const inlineValue = element.style.getPropertyValue(variableName);
        if (inlineValue) {
            return inlineValue.trim();
        }
        
        return null;
    }

    setCSSVariable(element, variableName, value) {
        element.style.setProperty(variableName, value);
    }

    countItemsInTimeline(line) {
        // Count timeline items associated with this line
        const timelineContainer = line.closest('.timeline-container');
        if (!timelineContainer) return 0;
        
        const items = timelineContainer.querySelectorAll('.timeline-item');
        return items.length;
    }

    // Utility methods for CSS variable access
    isValidLine(lineId) {
        if (lineId < 0 || lineId >= this.lines.length) {
            console.error(`Invalid line ID: ${lineId}. Available lines: 0-${this.lines.length - 1}`);
            return false;
        }
        return true;
    }
}


// Initialize the timeline controller
const timelineController = new TimelineController();

console.log('Timeline.js loaded successfully!');
