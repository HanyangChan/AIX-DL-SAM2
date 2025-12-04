import React, { useEffect, useRef } from 'react';

const ResultDisplay = ({ image, result, onReset }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        if (!image || !result || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const img = new Image();

        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;

            // Draw original image
            ctx.drawImage(img, 0, 0);

            // Draw items (Masks, Contours, Labels)
            result.food_items.forEach(item => {
                // Mask Overlay removed as per user request
                // We only draw annotations (contours and labels)
                drawAnnotations(ctx, item);
            });
        };
        img.src = URL.createObjectURL(image);

    }, [image, result]);

    const drawAnnotations = (ctx, item) => {
        // 2. Red Contour
        if (item.contours && item.contours.length > 0) {
            ctx.beginPath();
            ctx.strokeStyle = '#ef4444'; // Red-500
            ctx.lineWidth = 4;
            item.contours.forEach((point, index) => {
                if (index === 0) ctx.moveTo(point[0], point[1]);
                else ctx.lineTo(point[0], point[1]);
            });
            ctx.closePath();
            ctx.stroke();
        }

        // 3. Label
        if (item.bbox && item.bbox.length === 4) {
            const [x1, y1, x2, y2] = item.bbox;
            const centerX = (x1 + x2) / 2;
            const centerY = (y1 + y2) / 2;

            ctx.font = '800 24px Inter, sans-serif'; // Extra Bold
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            const text = `${item.label} (${item.calories} kcal)`;
            const metrics = ctx.measureText(text);
            const padding = 16;
            const boxHeight = 48;

            // Draw White Background
            ctx.fillStyle = '#ffffff'; // Pure White
            ctx.beginPath();
            ctx.roundRect(
                centerX - metrics.width / 2 - padding,
                centerY - boxHeight / 2,
                metrics.width + padding * 2,
                boxHeight,
                8
            );
            ctx.fill();

            // Draw Black Text
            ctx.fillStyle = '#000000'; // Pure Black
            ctx.fillText(text, centerX, centerY);
        }
    };

    return (
        <div className="card">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '2rem' }}>

                {/* Image Area */}
                <div style={{ position: 'relative', minHeight: '400px', background: '#000', borderRadius: '0.5rem', overflow: 'hidden' }}>
                    <canvas
                        ref={canvasRef}
                        style={{ width: '100%', height: 'auto', display: 'block' }}
                    />
                </div>

                {/* Info Area */}
                <div>
                    <h2 className="title" style={{ fontSize: '2rem' }}>Results</h2>

                    <div style={{ marginBottom: '2rem' }}>
                        <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Total Calories</div>
                        <div style={{ fontSize: '3rem', fontWeight: '800', color: 'var(--accent)' }}>
                            {result.calories} <span style={{ fontSize: '1rem' }}>kcal</span>
                        </div>
                    </div>

                    <h3 style={{ marginBottom: '1rem' }}>Detected Items</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {result.food_items.map((item, idx) => (
                            <div key={idx} style={{
                                background: 'rgba(255,255,255,0.05)',
                                padding: '1rem',
                                borderRadius: '0.5rem',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center'
                            }}>
                                <span style={{ fontWeight: '600' }}>{item.label || `Item ${idx + 1}`}</span>
                                <span style={{ color: 'var(--accent)' }}>{item.calories} kcal</span>
                            </div>
                        ))}
                    </div>

                    <button
                        className="btn"
                        style={{ width: '100%', marginTop: '2rem', background: 'var(--bg-primary)', border: '1px solid var(--text-secondary)' }}
                        onClick={onReset}
                    >
                        Analyze Another
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ResultDisplay;
