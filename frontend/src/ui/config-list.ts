// T037: Config selector UI component

interface Config {
  id: string;
  name: string;
  path: string;
  fluid_type: string;
  particle_count_estimate: number;
}

interface ConfigsResponse {
  configs: Config[];
}

type SelectCallback = (configName: string) => void;

/**
 * Configuration selector UI component
 *
 * Fetches available simulation configs from the server
 * and displays them as a selectable list.
 */
export class ConfigList {
  private container: HTMLElement;
  private selectCallbacks: SelectCallback[] = [];
  private configs: Config[] = [];
  private selectedConfig: string | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.render();
  }

  /**
   * Register callback for config selection
   */
  onSelect(cb: SelectCallback): void {
    this.selectCallbacks.push(cb);
  }

  /**
   * Fetch configs from server and refresh UI
   */
  async refresh(): Promise<void> {
    try {
      const response = await fetch('/api/configs');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: ConfigsResponse = await response.json();
      this.configs = data.configs;
      this.render();
    } catch (error) {
      console.error('Failed to fetch configs:', error);
      this.renderError('Failed to load configurations');
    }
  }

  /**
   * Render the config list UI
   */
  private render(): void {
    this.container.innerHTML = `
      <div style="padding: 12px; background: #2a2a2a; border-radius: 4px;">
        <h3 style="margin: 0 0 12px 0; font-size: 14px; color: #fff;">
          Simulation Configs
        </h3>
        <div id="config-list-items">
          ${this.configs.length === 0
            ? '<p style="color: #888; font-size: 12px;">Loading...</p>'
            : this.renderConfigItems()
          }
        </div>
      </div>
    `;

    // Attach event listeners
    this.configs.forEach((config) => {
      const button = document.getElementById(`config-${config.id}`);
      if (button) {
        button.addEventListener('click', () => this.handleSelect(config.id));
      }
    });
  }

  /**
   * Render individual config items
   */
  private renderConfigItems(): string {
    return this.configs
      .map((config) => {
        const isSelected = this.selectedConfig === config.id;
        return `
          <button
            id="config-${config.id}"
            style="
              display: block;
              width: 100%;
              padding: 8px;
              margin-bottom: 6px;
              background: ${isSelected ? '#2196F3' : '#1a1a1a'};
              color: #fff;
              border: 1px solid ${isSelected ? '#2196F3' : '#444'};
              border-radius: 3px;
              cursor: pointer;
              font-size: 12px;
              text-align: left;
              transition: all 0.2s;
            "
            onmouseover="this.style.background='${isSelected ? '#1976D2' : '#333'}'"
            onmouseout="this.style.background='${isSelected ? '#2196F3' : '#1a1a1a'}'"
          >
            <div style="font-weight: 500;">${config.name}</div>
            <div style="font-size: 10px; color: #aaa; margin-top: 2px;">
              ${config.fluid_type} • ~${this.formatCount(config.particle_count_estimate)} particles
            </div>
          </button>
        `;
      })
      .join('');
  }

  /**
   * Handle config selection
   */
  private handleSelect(configName: string): void {
    this.selectedConfig = configName;
    this.render();
    this.selectCallbacks.forEach((cb) => cb(configName));
  }

  /**
   * Render error message
   */
  private renderError(message: string): void {
    this.container.innerHTML = `
      <div style="padding: 12px; background: #2a2a2a; border-radius: 4px;">
        <p style="color: #f44336; font-size: 12px; margin: 0;">${message}</p>
      </div>
    `;
  }

  /**
   * Format particle count for display
   */
  private formatCount(count: number): string {
    if (count >= 1000000) {
      return `${(count / 1000000).toFixed(1)}M`;
    } else if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}K`;
    } else {
      return count.toString();
    }
  }
}
