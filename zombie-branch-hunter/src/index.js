'use strict';

require('dotenv').config();

const { GitHubClient } = require('./github');
const { AIClient } = require('./ai');

const MS_PER_DAY = 1000 * 60 * 60 * 24;

/**
 * Читает конфиг из переменных окружения (и из GitHub Action inputs,
 * которые GitHub прокидывает как INPUT_<NAME> в верхнем регистре).
 */
function loadConfig() {
  const env = process.env;

  const githubToken = env.GITHUB_TOKEN || env['INPUT_GITHUB-TOKEN'];
  const repoFull = env.GITHUB_REPOSITORY || '';
  const [owner, repo] = repoFull.split('/');

  const geminiApiKey = env.GEMINI_API_KEY || env['INPUT_GEMINI-API-KEY'];
  const geminiModel = env.GEMINI_MODEL || env['INPUT_GEMINI-MODEL'] || undefined;

  const inactiveDaysThreshold = parseIntSafe(
    env.INACTIVE_DAYS_THRESHOLD || env['INPUT_INACTIVE-DAYS-THRESHOLD'],
    30
  );
  const dedupWindowDays = parseIntSafe(
    env.DEDUP_WINDOW_DAYS || env['INPUT_DEDUP-WINDOW-DAYS'],
    7
  );
  const requestDelayMs = parseIntSafe(
    env.GEMINI_REQUEST_DELAY_MS || env['INPUT_GEMINI-REQUEST-DELAY-MS'],
    4500
  );
  const maxRetries = parseIntSafe(
    env.GEMINI_MAX_RETRIES || env['INPUT_GEMINI-MAX-RETRIES'],
    5
  );

  const dryRun = parseBoolSafe(env.DRY_RUN || env['INPUT_DRY-RUN'], false);

  return {
    githubToken,
    owner,
    repo,
    geminiApiKey,
    geminiModel,
    inactiveDaysThreshold,
    dedupWindowDays,
    requestDelayMs,
    maxRetries,
    dryRun,
  };
}

function parseIntSafe(value, fallback) {
  const n = parseInt(value, 10);
  return Number.isFinite(n) ? n : fallback;
}

function parseBoolSafe(value, fallback) {
  if (value === undefined || value === null || value === '') return fallback;
  return String(value).toLowerCase() === 'true';
}

function daysBetween(date, now) {
  if (!date) return null;
  return Math.floor((now.getTime() - date.getTime()) / MS_PER_DAY);
}

/**
 * Собирает контекст (коммиты, изменённые файлы) для одной ветки/PR,
 * не завязываясь на то, есть ли у неё открытый PR.
 */
async function buildCandidateContext(github, { branchName, defaultBranch }) {
  const [commitMessages, changedFiles] = await Promise.all([
    github.getRecentCommitMessages(branchName, 3),
    branchName === defaultBranch
      ? Promise.resolve([])
      : github.getChangedFileNames(defaultBranch, branchName),
  ]);
  return { commitMessages, changedFiles };
}

/**
 * Основной сценарий: находит зомби-ветки и зомби-PR, генерирует сообщения,
 * постит их (с защитой от дублей и уважением к rate limit).
 */
async function run() {
  const config = loadConfig();
  const now = new Date();

  console.log('=== Zombie Branch Hunter ===');
  console.log(`Репозиторий: ${config.owner}/${config.repo}`);
  console.log(`Порог неактивности: ${config.inactiveDaysThreshold} дней`);
  console.log(`Dry run: ${config.dryRun ? 'да (ничего не будет запощено)' : 'нет'}`);
  console.log('');

  let github;
  try {
    github = new GitHubClient({
      token: config.githubToken,
      owner: config.owner,
      repo: config.repo,
      dryRun: config.dryRun,
    });
  } catch (err) {
    console.error(`[fatal] Ошибка инициализации GitHub-клиента: ${err.message}`);
    process.exitCode = 1;
    return;
  }

  let ai;
  try {
    ai = new AIClient({
      apiKey: config.geminiApiKey,
      model: config.geminiModel,
      requestDelayMs: config.requestDelayMs,
      maxRetries: config.maxRetries,
    });
  } catch (err) {
    console.error(`[fatal] Ошибка инициализации Gemini-клиента: ${err.message}`);
    process.exitCode = 1;
    return;
  }

  let defaultBranch;
  try {
    defaultBranch = await github.getDefaultBranch();
  } catch (err) {
    console.error(
      `[fatal] Не удалось получить репозиторий ${config.owner}/${config.repo}. ` +
        `Проверь, что репозиторий существует и токен имеет доступ к нему. Ошибка: ${err.message}`
    );
    process.exitCode = 1;
    return;
  }
  console.log(`Основная ветка: ${defaultBranch}`);

  // --- 1. Собираем все ветки и все открытые PR ---
  let branches = [];
  let openPRs = [];
  try {
    [branches, openPRs] = await Promise.all([
      github.listBranches(),
      github.listOpenPullRequests(),
    ]);
  } catch (err) {
    console.error(`[fatal] Не удалось получить список веток/PR: ${err.message}`);
    process.exitCode = 1;
    return;
  }

  if (branches.length === 0) {
    console.log('В репозитории нет веток (возможно, он пустой). Завершаю работу.');
    return;
  }

  console.log(`Найдено веток: ${branches.length}`);
  console.log(`Найдено открытых PR: ${openPRs.length}`);
  console.log('');

  // Карта branchName -> PR, чтобы быстро понимать, есть ли у ветки открытый PR
  const prByBranchName = new Map();
  for (const pr of openPRs) {
    if (pr.head && pr.head.ref) {
      prByBranchName.set(pr.head.ref, pr);
    }
  }

  // --- 2 и 3. Определяем дату последнего коммита и фильтруем "зомби" ---
  const zombieCandidates = [];

  for (const branch of branches) {
    if (branch.name === defaultBranch) continue; // основную ветку никогда не трогаем

    const lastCommit = await github.getLastCommitInfo(branch.name);
    if (!lastCommit || !lastCommit.date) {
      console.warn(`[skip] Нет данных о последнем коммите для ветки "${branch.name}", пропускаю`);
      continue;
    }

    const inactiveDays = daysBetween(lastCommit.date, now);
    if (inactiveDays === null || inactiveDays <= config.inactiveDaysThreshold) {
      continue;
    }

    zombieCandidates.push({
      branchName: branch.name,
      lastCommitDate: lastCommit.date,
      daysInactive: inactiveDays,
      pr: prByBranchName.get(branch.name) || null,
    });
  }

  if (zombieCandidates.length === 0) {
    console.log(
      `Заброшенных веток не найдено (порог: ${config.inactiveDaysThreshold} дней). Всё чисто!`
    );
    return;
  }

  console.log(`Найдено зомби-кандидатов: ${zombieCandidates.length}`);
  zombieCandidates.forEach((c) => {
    console.log(
      `  - ${c.branchName} (${c.daysInactive} дн., ${c.pr ? `PR #${c.pr.number}` : 'без PR'})`
    );
  });
  console.log('');

  // --- 4-6. Для каждого кандидата собираем контекст и генерируем сообщение ---
  const prResults = []; // { pr, message } — постим сразу в комментарий
  const branchOnlyResults = []; // { branchName, daysInactive, message } — идут в сводный Issue
  const errors = [];

  for (const candidate of zombieCandidates) {
    console.log(`[обработка] ${candidate.branchName}...`);

    let mergeableState = null;
    let prNumber = null;
    if (candidate.pr) {
      prNumber = candidate.pr.number;
      try {
        const fullPR = await github.getPullRequest(prNumber);
        mergeableState = fullPR.mergeable_state || null;
      } catch (err) {
        console.warn(
          `[warn] Не удалось получить mergeable_state для PR #${prNumber}: ${err.message}`
        );
      }
    }

    let contextData;
    try {
      contextData = await buildCandidateContext(github, {
        branchName: candidate.branchName,
        defaultBranch,
      });
    } catch (err) {
      console.error(
        `[error] Не удалось собрать контекст для "${candidate.branchName}": ${err.message}`
      );
      errors.push({ branchName: candidate.branchName, error: err.message });
      continue;
    }

    const hasConflicts = mergeableState === 'dirty';

    let message;
    try {
      message = await ai.generateMessage({
        branchName: candidate.branchName,
        commitMessages: contextData.commitMessages,
        changedFiles: contextData.changedFiles,
        daysInactive: candidate.daysInactive,
        hasConflicts,
      });
    } catch (err) {
      console.error(
        `[error] Gemini API не смог сгенерировать сообщение для "${candidate.branchName}": ${err.message}`
      );
      errors.push({ branchName: candidate.branchName, error: err.message });
      continue;
    }

    if (candidate.pr) {
      prResults.push({ pr: candidate.pr, message, daysInactive: candidate.daysInactive });
    } else {
      branchOnlyResults.push({
        branchName: candidate.branchName,
        daysInactive: candidate.daysInactive,
        message,
      });
    }
  }

  console.log('');

  // --- 7. Постим комментарии в PR (с защитой от дублей) ---
  let postedPRComments = 0;
  let skippedPRComments = 0;

  for (const { pr, message } of prResults) {
    try {
      const alreadyCommented = await github.hasBotCommentedRecently(
        pr.number,
        config.dedupWindowDays
      );
      if (alreadyCommented) {
        console.log(
          `[skip] PR #${pr.number}: бот уже комментировал за последние ${config.dedupWindowDays} дн., пропускаю`
        );
        skippedPRComments += 1;
        continue;
      }

      await github.postComment(pr.number, message);
      console.log(`[posted] Комментарий в PR #${pr.number}`);
      postedPRComments += 1;
    } catch (err) {
      console.error(`[error] Не удалось запостить комментарий в PR #${pr.number}: ${err.message}`);
      errors.push({ branchName: pr.head.ref, error: err.message });
    }
  }

  // --- 8. Собираем сводку по веткам без PR в Issue ---
  let reportIssueUrl = null;
  if (branchOnlyResults.length > 0) {
    try {
      const { title, body } = buildReportIssueContent(branchOnlyResults, now);
      const issue = await github.upsertReportIssue(title, body);
      reportIssueUrl = issue && issue.html_url ? issue.html_url : null;
      console.log(
        `[posted] Сводный Issue обновлён: ${reportIssueUrl || '(dry-run, ссылки нет)'}`
      );
    } catch (err) {
      console.error(`[error] Не удалось создать/обновить сводный Issue: ${err.message}`);
      errors.push({ branchName: '(report issue)', error: err.message });
    }
  }

  // --- Итоговая сводка в лог ---
  console.log('');
  console.log('=== Готово ===');
  console.log(`Обработано кандидатов: ${zombieCandidates.length}`);
  console.log(`Запощено комментариев в PR: ${postedPRComments}`);
  console.log(`Пропущено (уже комментировали недавно): ${skippedPRComments}`);
  console.log(`Веток без PR в сводке: ${branchOnlyResults.length}`);
  if (errors.length > 0) {
    console.log(`Ошибок при обработке: ${errors.length}`);
    errors.forEach((e) => console.log(`  - ${e.branchName}: ${e.error}`));
  }
}

/**
 * Формирует заголовок и тело сводного Issue "Zombie Branches Report".
 */
function buildReportIssueContent(branchOnlyResults, now) {
  const dateStr = now.toISOString().slice(0, 10);
  const title = `🧟 Zombie Branches Report — ${dateStr}`;

  const lines = branchOnlyResults.map((r) => {
    return `### \`${r.branchName}\` (${r.daysInactive} дн. без активности)\n\n${r.message}\n`;
  });

  const body =
    `Еженедельный отчёт о ветках без открытого PR, которые не обновлялись дольше порога.\n\n` +
    `Обновлено: ${now.toISOString()}\n\n---\n\n` +
    lines.join('\n---\n\n');

  return { title, body };
}

// Запускаем только если файл вызван напрямую (не require()), чтобы его можно
// было импортировать в тестах без побочных эффектов.
if (require.main === module) {
  run().catch((err) => {
    console.error(`[fatal] Необработанная ошибка: ${err && err.stack ? err.stack : err}`);
    process.exitCode = 1;
  });
}

module.exports = { run, loadConfig, buildReportIssueContent };
