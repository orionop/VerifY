<script>
  // State variables
  let title = '';
  let isVerifying = false;
  let verificationResult = null;
  let error = null;
  let resubmitTitle = '';

  // Simulate API call to verify title
  async function verifyTitle(titleToVerify) {
    isVerifying = true;
    error = null;
    
    try {
      // In a real application, this would be a fetch to your backend API:
      // const response = await fetch('/verify-title', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ title: titleToVerify })
      // });
      // const data = await response.json();
      
      // Simulating API response with dummy data
      await new Promise(resolve => setTimeout(resolve, 800)); // Simulate network delay
      
      // Dummy response data
      const data = {
        similarTitles: [
          { title: 'System for Title Verification', similarity: 82 },
          { title: 'Title Verification Method', similarity: 65 },
          { title: 'Verification of Document Titles', similarity: 58 }
        ],
        matchScore: titleToVerify.length > 5 ? 78 : 35,
        disallowedWords: titleToVerify.toLowerCase().includes('patent') ? ['patent'] : [],
        status: titleToVerify.length > 5 ? 'Rejected' : 'Accepted',
        approvalProbability: titleToVerify.length > 10 ? 24 : 84,
      };
      
      verificationResult = data;
    } catch (err) {
      error = 'An error occurred while verifying the title. Please try again.';
      console.error('Verification error:', err);
    } finally {
      isVerifying = false;
    }
  }

  function handleSubmit() {
    if (title.trim()) {
      verifyTitle(title);
    }
  }

  function handleResubmit() {
    if (resubmitTitle.trim()) {
      title = resubmitTitle;
      verifyTitle(resubmitTitle);
      resubmitTitle = '';
    }
  }
  
  function handleKeydown(event) {
    if (event.key === 'Enter' && title.trim()) {
      handleSubmit();
    }
  }
</script>

<main>
  <div class="container">
    <header>
      <h1>Title Verification</h1>
      <div class="subtitle">Press Registrar General of India</div>
    </header>

    <section class="search-section">
      <div class="search-container">
        <input
          type="text"
          bind:value={title}
          placeholder="Enter your proposed title"
          disabled={isVerifying}
          on:keydown={handleKeydown}
          class="search-input"
        />
        <button 
          on:click={handleSubmit}
          disabled={!title.trim() || isVerifying}
          class="search-button"
        >
          {isVerifying ? 'Verifying...' : 'Verify'}
        </button>
      </div>
    </section>

    {#if error}
      <div class="alert error">
        <p>{error}</p>
      </div>
    {/if}

    {#if verificationResult}
      <section class="results-section">
        <div class="result-card status-card {verificationResult.status.toLowerCase()}">
          <div class="card-content">
            <div class="status-badge">{verificationResult.status}</div>
            <div class="probability">
              <div class="probability-label">Approval probability</div>
              <div class="probability-value">{verificationResult.approvalProbability}%</div>
            </div>
          </div>
        </div>
        
        <div class="result-metrics">
          <div class="metric-card">
            <h3>Match Score</h3>
            <div class="score-indicator">
              <div class="score-bar" style="width: {verificationResult.matchScore}%"></div>
              <span>{verificationResult.matchScore}%</span>
            </div>
          </div>

          {#if verificationResult.disallowedWords.length > 0}
            <div class="metric-card disallowed">
              <h3>Disallowed Words</h3>
              <ul class="tag-list">
                {#each verificationResult.disallowedWords as word}
                  <li class="tag">{word}</li>
                {/each}
              </ul>
            </div>
          {/if}
        </div>

        {#if verificationResult.similarTitles.length > 0}
          <div class="result-card">
            <h3>Similar Titles</h3>
            <ul class="similar-titles">
              {#each verificationResult.similarTitles as item}
                <li>
                  <div class="title-text">{item.title}</div>
                  <div class="similarity-badge" class:high={item.similarity > 80} class:medium={item.similarity > 60 && item.similarity <= 80} class:low={item.similarity <= 60}>
                    {item.similarity}%
                  </div>
                </li>
              {/each}
            </ul>
          </div>
        {/if}

        {#if verificationResult.status === 'Rejected'}
          <div class="result-card resubmit-card">
            <h3>Try a Different Title</h3>
            <div class="resubmit-form">
              <input
                type="text"
                bind:value={resubmitTitle}
                placeholder="Enter a revised title"
                disabled={isVerifying}
                class="resubmit-input"
              />
              <button 
                on:click={handleResubmit}
                disabled={!resubmitTitle.trim() || isVerifying}
                class="resubmit-button"
              >
                {isVerifying ? 'Verifying...' : 'Verify'}
              </button>
            </div>
          </div>
        {/if}
      </section>
    {/if}
  </div>
</main>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    background-color: #f7f7f7;
    font-family: Circular, -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif;
    color: #222222;
    line-height: 1.6;
  }

  main {
    min-height: 100vh;
    padding: 2rem 1rem;
  }

  .container {
    max-width: 720px;
    margin: 0 auto;
  }

  header {
    text-align: center;
    margin-bottom: 2.5rem;
  }

  h1 {
    font-size: 2rem;
    font-weight: 600;
    margin: 0;
    color: #222222;
  }
  
  .subtitle {
    color: #717171;
    font-size: 1rem;
    margin-top: 0.25rem;
  }

  h3 {
    font-size: 1.125rem;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 1rem;
    color: #222222;
  }

  .search-section {
    margin-bottom: 2rem;
  }

  .search-container {
    display: flex;
    border-radius: 40px;
    background: white;
    border: 1px solid #DDDDDD;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
    overflow: hidden;
    transition: box-shadow 0.2s ease;
  }

  .search-container:focus-within {
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.12);
  }

  .search-input {
    flex: 1;
    border: none;
    padding: 16px 24px;
    font-size: 16px;
    outline: none;
    color: #222222;
  }

  .search-button {
    background: #FF385C;
    color: white;
    border: none;
    padding: 0 28px;
    font-size: 16px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s ease;
  }

  .search-button:hover {
    background-color: #E61E4D;
  }

  .search-button:disabled {
    background-color: #DDDDDD;
    cursor: not-allowed;
  }

  .results-section {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }

  .result-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
  }

  .result-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.25rem;
  }

  .metric-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
  }

  .status-card {
    border-left: 4px solid #DDDDDD;
  }

  .status-card.accepted {
    border-left-color: #00A699;
  }

  .status-card.rejected {
    border-left-color: #FF385C;
  }

  .card-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .status-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 30px;
    font-weight: 500;
    font-size: 14px;
  }

  .accepted .status-badge {
    background-color: #ECFDF3;
    color: #00A699;
  }

  .rejected .status-badge {
    background-color: #FFF1F1;
    color: #FF385C;
  }

  .probability {
    text-align: right;
  }

  .probability-label {
    font-size: 14px;
    color: #717171;
    margin-bottom: 4px;
  }

  .probability-value {
    font-size: 24px;
    font-weight: 600;
  }

  .accepted .probability-value {
    color: #00A699;
  }

  .rejected .probability-value {
    color: #FF385C;
  }

  .score-indicator {
    height: 8px;
    width: 100%;
    background-color: #EBEBEB;
    border-radius: 4px;
    position: relative;
    margin-bottom: 8px;
    overflow: hidden;
  }

  .score-bar {
    height: 100%;
    background-color: #FF385C;
    border-radius: 4px;
    transition: width 0.5s ease;
  }

  .score-indicator span {
    font-size: 14px;
    font-weight: 500;
    color: #717171;
  }

  .tag-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .tag {
    background-color: #FFF1F1;
    color: #FF385C;
    padding: 6px 12px;
    border-radius: 30px;
    font-size: 14px;
    font-weight: 500;
  }

  .similar-titles {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .similar-titles li {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid #EBEBEB;
  }

  .similar-titles li:last-child {
    border-bottom: none;
  }

  .title-text {
    font-weight: 500;
  }

  .similarity-badge {
    padding: 4px 10px;
    border-radius: 30px;
    font-size: 14px;
    font-weight: 500;
  }

  .similarity-badge.high {
    background-color: #FFF1F1;
    color: #FF385C;
  }

  .similarity-badge.medium {
    background-color: #FFF7E6;
    color: #FFA03F;
  }

  .similarity-badge.low {
    background-color: #ECFDF3;
    color: #00A699;
  }

  .resubmit-form {
    display: flex;
    gap: 12px;
  }

  .resubmit-input {
    flex: 1;
    border: 1px solid #DDDDDD;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 16px;
    outline: none;
    transition: border-color 0.2s ease;
  }

  .resubmit-input:focus {
    border-color: #B0B0B0;
  }

  .resubmit-button {
    background: #222222;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0 20px;
    font-size: 16px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s ease;
  }

  .resubmit-button:hover {
    background-color: #000000;
  }

  .resubmit-button:disabled {
    background-color: #DDDDDD;
    cursor: not-allowed;
  }

  .alert {
    padding: 16px;
    border-radius: 8px;
    margin-bottom: 20px;
  }

  .alert.error {
    background-color: #FFF1F1;
    color: #FF385C;
  }

  .alert p {
    margin: 0;
  }

  @media (max-width: 640px) {
    .card-content {
      flex-direction: column;
      align-items: flex-start;
      gap: 16px;
    }

    .probability {
      text-align: left;
    }

    .resubmit-form {
      flex-direction: column;
    }
  }
</style>
