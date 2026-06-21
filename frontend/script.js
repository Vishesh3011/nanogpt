// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => {
            c.classList.add('hidden');
            c.classList.remove('active');
        });
        tab.classList.add('active');
        const target = document.getElementById(tab.dataset.tab);
        target.classList.remove('hidden');
        target.classList.add('active');
    });
});

// Theme toggle
const toggle = document.getElementById('theme-toggle');
toggle.addEventListener('click', () => {
    const html = document.documentElement;
    const isDark = html.getAttribute('data-theme') === 'dark';
    html.setAttribute('data-theme', isDark ? 'light' : 'dark');
    toggle.textContent = isDark ? '☾ Dark' : '☀ Light';
});

// Generic generate function
async function generate(endpoint, prompt, maxTokens, temperature, topK, outputEl, errorEl, btn) {
    errorEl.classList.add('hidden');
    outputEl.classList.add('hidden');
    btn.disabled = true;
    btn.textContent = 'Generating...';

    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                max_new_tokens: parseInt(maxTokens),
                temperature: parseFloat(temperature),
                top_k: parseInt(topK),
            }),
        });

        const data = await res.json();

        if (!res.ok) {
            // FastAPI returns { detail: "..." } for HTTP errors
            errorEl.textContent = data.detail || `Error ${res.status}`;
            errorEl.classList.remove('hidden');
            return;
        }

        outputEl.textContent = data.generated_text;
        outputEl.classList.remove('hidden');

    } catch (err) {
        errorEl.textContent = `Network error: ${err.message}`;
        errorEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = btn.id === 'story-btn' ? 'Generate Story' : 'Generate Code';
    }
}

// Story button
document.getElementById('story-btn').addEventListener('click', () => {
    generate(
        '/generate/story',
        document.getElementById('story-prompt').value,
        document.getElementById('story-tokens').value,
        document.getElementById('story-temp').value,
        200,   // top_k fixed for stories
        document.getElementById('story-output'),
        document.getElementById('story-error'),
        document.getElementById('story-btn'),
    );
});

// Code button
document.getElementById('code-btn').addEventListener('click', () => {
    generate(
        '/generate/code',
        document.getElementById('code-prompt').value,
        document.getElementById('code-tokens').value,
        document.getElementById('code-temp').value,
        40,    // lower top_k for code (more deterministic)
        document.getElementById('code-output'),
        document.getElementById('code-error'),
        document.getElementById('code-btn'),
    );
});