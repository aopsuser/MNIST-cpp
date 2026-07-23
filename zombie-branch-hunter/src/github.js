'use strict';

const { Octokit } = require('@octokit/rest');

const BOT_SIGNATURE = '<!-- zombie-branch-hunter:bot-comment -->';
const REPORT_ISSUE_MARKER = '<!-- zombie-branch-hunter:report-issue -->';

/**
 * Тонкая обёртка над @octokit/rest, заточенная под нужды этого Action.
 * Всё, что говорит с GitHub API, живёт здесь — это единственное место,
 * которое нужно будет менять, если поменяется формат GitHub API.
 */
class GitHubClient {
  /**
   * @param {object} opts
   * @param {string} opts.token - GITHUB_TOKEN или PAT
   * @param {string} opts.owner
   * @param {string} opts.repo
   * @param {boolean} [opts.dryRun=false] - если true, ничего не постит, только логирует
   */
  constructor({ token, owner, repo, dryRun = false }) {
    if (!token) {
      throw new Error(
        'GitHub token не задан. Внутри GitHub Action он должен приходить из ${{ secrets.GITHUB_TOKEN }}, ' +
          'при локальном запуске — из переменной GITHUB_TOKEN в .env'
      );
    }
    if (!owner || !repo) {
      throw new Error(
        'Не заданы owner/repo. Внутри Action они берутся из GITHUB_REPOSITORY, ' +
          'при локальном запуске — задай GITHUB_REPOSITORY=owner/repo в .env'
      );
    }

    this.octokit = new Octokit({ auth: token });
    this.owner = owner;
    this.repo = repo;
    this.dryRun = dryRun;
  }

  /**
   * Возвращает список всех веток репозитория (с пагинацией).
   * @returns {Promise<Array<{name: string, commitSha: string}>>}
   */
  async listBranches() {
    const branches = await this.octokit.paginate(this.octokit.rest.repos.listBranches, {
      owner: this.owner,
      repo: this.repo,
      per_page: 100,
    });

    return branches.map((b) => ({
      name: b.name,
      commitSha: b.commit.sha,
    }));
  }

  /**
   * Возвращает список открытых PR (с пагинацией).
   * @returns {Promise<Array<object>>} сырые объекты PR из GitHub API
   */
  async listOpenPullRequests() {
    return this.octokit.paginate(this.octokit.rest.pulls.list, {
      owner: this.owner,
      repo: this.repo,
      state: 'open',
      per_page: 100,
    });
  }

  /**
   * Получает подробности одного PR (нужно отдельно, чтобы узнать mergeable_state —
   * в списке PR это поле не отдаётся GitHub API).
   * @param {number} pullNumber
   */
  async getPullRequest(pullNumber) {
    const { data } = await this.octokit.rest.pulls.get({
      owner: this.owner,
      repo: this.repo,
      pull_number: pullNumber,
    });
    return data;
  }

  /**
   * Дата последнего коммита в ветке/PR по имени ветки или SHA.
   * @param {string} ref - имя ветки или SHA
   * @returns {Promise<{date: Date, sha: string, message: string} | null>}
   */
  async getLastCommitInfo(ref) {
    try {
      const { data } = await this.octokit.rest.repos.listCommits({
        owner: this.owner,
        repo: this.repo,
        sha: ref,
        per_page: 1,
      });

      if (!data || data.length === 0) {
        return null;
      }

      const commit = data[0];
      const dateStr =
        commit.commit.committer?.date || commit.commit.author?.date || null;

      return {
        date: dateStr ? new Date(dateStr) : null,
        sha: commit.sha,
        message: commit.commit.message,
      };
    } catch (err) {
      // Пустой репозиторий / ветка без коммитов / удалённая ветка — не фатально
      console.warn(`[github] Не удалось получить коммиты для "${ref}": ${err.message}`);
      return null;
    }
  }

  /**
   * Последние N коммит-мессаджей ветки.
   * @param {string} ref
   * @param {number} count
   * @returns {Promise<string[]>}
   */
  async getRecentCommitMessages(ref, count = 3) {
    try {
      const { data } = await this.octokit.rest.repos.listCommits({
        owner: this.owner,
        repo: this.repo,
        sha: ref,
        per_page: count,
      });
      return data.map((c) => firstLine(c.commit.message));
    } catch (err) {
      console.warn(`[github] Не удалось получить коммит-мессаджи для "${ref}": ${err.message}`);
      return [];
    }
  }

  /**
   * Список имён изменённых файлов между base (обычно main) и head (ветка).
   * Использует GET /compare — без полного diff, только имена файлов,
   * чтобы не раздувать промпт для LLM.
   * @param {string} base
   * @param {string} head
   * @returns {Promise<string[]>}
   */
  async getChangedFileNames(base, head) {
    try {
      const { data } = await this.octokit.rest.repos.compareCommits({
        owner: this.owner,
        repo: this.repo,
        base,
        head,
      });
      const files = data.files || [];
      return files.map((f) => f.filename);
    } catch (err) {
      // compare может упасть, если веток разошлись слишком сильно
      // или base/head не существуют — не фатально, просто пустой список
      console.warn(
        `[github] Не удалось сравнить "${base}...${head}": ${err.message}`
      );
      return [];
    }
  }

  /**
   * Имя основной ветки репозитория (main/master/что угодно).
   * @returns {Promise<string>}
   */
  async getDefaultBranch() {
    const { data } = await this.octokit.rest.repos.get({
      owner: this.owner,
      repo: this.repo,
    });
    return data.default_branch;
  }

  /**
   * Проверяет, комментировал ли уже бот этот PR/issue за последние `withinDays` дней.
   * Используется как защита от дублирующих комментариев.
   * @param {number} issueOrPrNumber
   * @param {number} withinDays
   * @returns {Promise<boolean>}
   */
  async hasBotCommentedRecently(issueOrPrNumber, withinDays = 7) {
    try {
      const comments = await this.octokit.paginate(
        this.octokit.rest.issues.listComments,
        {
          owner: this.owner,
          repo: this.repo,
          issue_number: issueOrPrNumber,
          per_page: 100,
        }
      );

      const cutoff = new Date();
      cutoff.setDate(cutoff.getDate() - withinDays);

      return comments.some((c) => {
        const isBotComment = (c.body || '').includes(BOT_SIGNATURE);
        if (!isBotComment) return false;
        const createdAt = new Date(c.created_at);
        return createdAt >= cutoff;
      });
    } catch (err) {
      console.warn(
        `[github] Не удалось проверить историю комментариев для #${issueOrPrNumber}: ${err.message}`
      );
      // При ошибке лучше перестраховаться и НЕ постить, чем задублировать
      return true;
    }
  }

  /**
   * Постит комментарий в PR (или issue — для GitHub API это один и тот же endpoint).
   * Автоматически добавляет служебную подпись-маркер для дедупликации.
   * @param {number} issueOrPrNumber
   * @param {string} body
   */
  async postComment(issueOrPrNumber, body) {
    const fullBody = `${body}\n\n${BOT_SIGNATURE}`;

    if (this.dryRun) {
      console.log(
        `[dry-run] Запостил бы комментарий в #${issueOrPrNumber}:\n---\n${fullBody}\n---`
      );
      return { dryRun: true };
    }

    const { data } = await this.octokit.rest.issues.createComment({
      owner: this.owner,
      repo: this.repo,
      issue_number: issueOrPrNumber,
      body: fullBody,
    });
    return data;
  }

  /**
   * Находит открытый issue-отчёт "Zombie Branches Report" (по служебному маркеру
   * в теле, а не по заголовку — так отчёт не потеряется, если поменять дату в заголовке).
   * @returns {Promise<object|null>}
   */
  async findExistingReportIssue() {
    try {
      const issues = await this.octokit.paginate(this.octokit.rest.issues.listForRepo, {
        owner: this.owner,
        repo: this.repo,
        state: 'open',
        labels: 'zombie-branch-hunter',
        per_page: 100,
      });

      return (
        issues.find((i) => (i.body || '').includes(REPORT_ISSUE_MARKER)) || null
      );
    } catch (err) {
      console.warn(`[github] Не удалось найти существующий report issue: ${err.message}`);
      return null;
    }
  }

  /**
   * Создаёт лейбл, если его ещё нет (нужен для findExistingReportIssue).
   */
  async ensureLabelExists() {
    if (this.dryRun) return;
    try {
      await this.octokit.rest.issues.getLabel({
        owner: this.owner,
        repo: this.repo,
        name: 'zombie-branch-hunter',
      });
    } catch (err) {
      if (err.status === 404) {
        try {
          await this.octokit.rest.issues.createLabel({
            owner: this.owner,
            repo: this.repo,
            name: 'zombie-branch-hunter',
            color: '6e40c9',
            description: 'Автоматически создано Zombie Branch Hunter',
          });
        } catch (createErr) {
          // Гонка условий/нет прав — не фатально, просто продолжаем без лейбла
          console.warn(`[github] Не удалось создать лейбл: ${createErr.message}`);
        }
      } else {
        console.warn(`[github] Не удалось проверить лейбл: ${err.message}`);
      }
    }
  }

  /**
   * Создаёт новый или обновляет существующий "Zombie Branches Report" issue.
   * @param {string} title
   * @param {string} body
   */
  async upsertReportIssue(title, body) {
    const fullBody = `${body}\n\n${REPORT_ISSUE_MARKER}`;

    if (this.dryRun) {
      console.log(
        `[dry-run] Создал/обновил бы Issue "${title}":\n---\n${fullBody}\n---`
      );
      return { dryRun: true };
    }

    await this.ensureLabelExists();
    const existing = await this.findExistingReportIssue();

    if (existing) {
      const { data } = await this.octokit.rest.issues.update({
        owner: this.owner,
        repo: this.repo,
        issue_number: existing.number,
        title,
        body: fullBody,
      });
      return data;
    }

    const { data } = await this.octokit.rest.issues.create({
      owner: this.owner,
      repo: this.repo,
      title,
      body: fullBody,
      labels: ['zombie-branch-hunter'],
    });
    return data;
  }
}

function firstLine(message) {
  return (message || '').split('\n')[0].trim();
}

module.exports = { GitHubClient, BOT_SIGNATURE, REPORT_ISSUE_MARKER };
