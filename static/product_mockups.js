const tabs = document.querySelectorAll('.concept-tab');
const screens = document.querySelectorAll('.screen');

tabs.forEach((tab) => tab.addEventListener('click', () => {
  tabs.forEach((item) => item.classList.remove('active'));
  screens.forEach((screen) => screen.classList.remove('active'));
  tab.classList.add('active');
  document.getElementById(tab.dataset.screen).classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}));

const modeContent = {
  quick: ['Quick mode', 'Reed is the safer play this week. Higgins is doubtful.', 'High confidence', '94%'],
  standard: ['Standard mode', 'Reed projects 8.4 points higher, driven by Higgins’ injury status and a favorable slot matchup.', '82% confidence', '82%'],
  expert: ['Expert mode', 'Median delta +8.4 (80% CI: +3.1 to +12.7); target-share prior 21.4%, matchup multiplier 1.08.', 'Model confidence 0.76', '76%']
};

document.querySelectorAll('.mode-card').forEach((card) => card.addEventListener('click', () => {
  document.querySelectorAll('.mode-card').forEach((item) => {
    item.classList.remove('selected');
    item.setAttribute('aria-checked', 'false');
  });
  card.classList.add('selected');
  card.setAttribute('aria-checked', 'true');
  const [name, description, confidence, width] = modeContent[card.dataset.mode];
  document.getElementById('modeName').textContent = name;
  document.getElementById('modeDescription').textContent = description;
  document.getElementById('confidenceText').textContent = confidence;
  document.getElementById('previewBar').style.width = width;
  document.querySelector('.mode-footer button').textContent = `Continue with ${name.replace(' mode', '')} →`;
}));
