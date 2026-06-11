// Parameter panel: editable simulation configuration form.
//
// A preset's raw JSON populates the form; `buildConfig()` produces the
// inline `config_json` payload for POST /api/simulations. The form is the
// single source of truth for what gets run.

type Json = Record<string, unknown>;

type WallValue = 'Wall' | 'Outflow' | 'Periodic' | 'Inflow';
const WALL_FACES = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'] as const;
type WallFace = (typeof WALL_FACES)[number];

type ObstacleKind = 'none' | 'sphere' | 'box' | 'cylinder' | 'stl';

interface FormState {
  name: string;
  fluidType: 'Water' | 'Air' | 'Mixed';
  solver: 'wcsph' | 'pcisph';
  backend: 'auto' | 'cpu' | 'gpu';
  spacing: number; // meters
  gravity: [number, number, number];
  viscosity: number;
  domainMin: [number, number, number];
  domainSize: [number, number, number]; // meters
  fillFrac: [number, number, number]; // 0..1 along each axis
  walls: Record<WallFace, WallValue>;
  inflowParams: Json | null; // preserved from preset when a face is Inflow
  obstacle: ObstacleKind;
  obstacleCenter: [number, number, number];
  obstacleRadius: number;
  obstacleHeight: number;
  obstacleAxis: 'x' | 'y' | 'z';
  obstacleBoxSize: [number, number, number];
  geometryFile: string | null; // preserved STL path from preset
  maxTime: number | null;
  // Passthrough values not exposed in the UI
  speedOfSound: number | undefined;
  cfl: number | undefined;
  initialTemperature: number | undefined;
  maxTimesteps: number | undefined;
}

function defaults(): FormState {
  return {
    name: 'Custom simulation',
    fluidType: 'Water',
    solver: 'wcsph',
    backend: 'auto',
    spacing: 0.0025,
    gravity: [0, -9.81, 0],
    viscosity: 0.001,
    domainMin: [0, 0, 0],
    domainSize: [0.12, 0.08, 0.06],
    fillFrac: [0.3, 0.75, 1.0],
    walls: {
      x_min: 'Wall',
      x_max: 'Wall',
      y_min: 'Wall',
      y_max: 'Outflow',
      z_min: 'Wall',
      z_max: 'Wall',
    },
    inflowParams: null,
    obstacle: 'none',
    obstacleCenter: [0.06, 0.04, 0.03],
    obstacleRadius: 0.008,
    obstacleHeight: 0.08,
    obstacleAxis: 'y',
    obstacleBoxSize: [0.016, 0.016, 0.016],
    geometryFile: null,
    maxTime: 5.0,
    speedOfSound: undefined,
    cfl: undefined,
    initialTemperature: undefined,
    maxTimesteps: undefined,
  };
}

export class ParamPanel {
  private container: HTMLElement;
  private state: FormState = defaults();
  private changeCallbacks: (() => void)[] = [];

  constructor(container: HTMLElement) {
    this.container = container;
    this.render();
  }

  onChange(cb: () => void): void {
    this.changeCallbacks.push(cb);
  }

  private emitChange(): void {
    this.updateEstimate();
    this.changeCallbacks.forEach((cb) => cb());
  }

  /** Populate the form from a preset's raw config JSON */
  loadConfig(json: Json): void {
    const s = defaults();

    s.name = (json.name as string) ?? 'Custom simulation';
    s.fluidType = (json.fluid_type as FormState['fluidType']) ?? 'Water';
    s.solver = (json.solver as FormState['solver']) ?? 'wcsph';
    s.backend = (json.backend as FormState['backend']) ?? 'auto';
    s.spacing = (json.particle_spacing as number) ?? 0.0025;
    s.gravity = ((json.gravity as number[]) ?? [0, -9.81, 0]) as [number, number, number];
    s.viscosity = (json.viscosity as number) ?? 0.001;

    const domain = json.domain as { min: number[]; max: number[] } | undefined;
    if (domain) {
      s.domainMin = domain.min as [number, number, number];
      s.domainSize = [
        domain.max[0] - domain.min[0],
        domain.max[1] - domain.min[1],
        domain.max[2] - domain.min[2],
      ];
    }

    const region = json.fluid_region as { min: number[]; max: number[] } | undefined;
    if (region) {
      s.fillFrac = [0, 1, 2].map((a) => {
        const frac = (region.max[a] - region.min[a]) / Math.max(1e-9, s.domainSize[a]);
        return Math.max(0.05, Math.min(1, frac));
      }) as [number, number, number];
    } else {
      s.fillFrac = [1, 1, 1];
    }

    const bc = json.boundary_conditions as Record<string, unknown> | undefined;
    if (bc) {
      for (const face of WALL_FACES) {
        const v = bc[face];
        if (typeof v === 'string') {
          s.walls[face] = v as WallValue;
        } else if (v && typeof v === 'object') {
          s.walls[face] = 'Inflow';
          s.inflowParams = (v as Json).Inflow as Json;
        }
      }
    }

    const geometry = json.geometry as Json | undefined;
    if (geometry && typeof geometry.type === 'string' && geometry.type !== 'none') {
      const kind = geometry.type as ObstacleKind;
      s.obstacle = kind;
      if (kind === 'sphere' || kind === 'cylinder') {
        s.obstacleCenter = geometry.center as [number, number, number];
        s.obstacleRadius = geometry.radius as number;
      }
      if (kind === 'cylinder') {
        s.obstacleHeight = geometry.height as number;
        s.obstacleAxis = ((geometry.axis as string) ?? 'y') as FormState['obstacleAxis'];
      }
      if (kind === 'box') {
        const min = geometry.min as number[];
        const max = geometry.max as number[];
        s.obstacleCenter = [
          (min[0] + max[0]) / 2,
          (min[1] + max[1]) / 2,
          (min[2] + max[2]) / 2,
        ];
        s.obstacleBoxSize = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
      }
    } else if (typeof json.geometry_file === 'string' && json.geometry_file) {
      // Preserve preset STL files (the "null obstacle" counts as none)
      if (json.geometry_file.includes('null-obstacle')) {
        s.obstacle = 'none';
      } else {
        s.obstacle = 'stl';
        s.geometryFile = json.geometry_file;
      }
    }

    s.maxTime = (json.max_time as number) ?? null;
    s.speedOfSound = json.speed_of_sound as number | undefined;
    s.cfl = json.cfl_number as number | undefined;
    s.initialTemperature = json.initial_temperature as number | undefined;
    s.maxTimesteps = json.max_timesteps as number | undefined;

    this.state = s;
    this.render();
    this.emitChange();
  }

  /** Produce the config_json payload for POST /api/simulations */
  buildConfig(): Json {
    const s = this.state;
    const domainMax: [number, number, number] = [
      s.domainMin[0] + s.domainSize[0],
      s.domainMin[1] + s.domainSize[1],
      s.domainMin[2] + s.domainSize[2],
    ];

    const config: Json = {
      name: s.name,
      fluid_type: s.fluidType,
      domain: { min: [...s.domainMin], max: domainMax },
      particle_spacing: s.spacing,
      gravity: [...s.gravity],
      viscosity: s.viscosity,
      backend: s.backend,
      solver: s.solver,
    };

    // Fluid region (only when not a full fill)
    if (s.fillFrac.some((f) => f < 0.999)) {
      config.fluid_region = {
        min: [...s.domainMin],
        max: [
          s.domainMin[0] + s.domainSize[0] * s.fillFrac[0],
          s.domainMin[1] + s.domainSize[1] * s.fillFrac[1],
          s.domainMin[2] + s.domainSize[2] * s.fillFrac[2],
        ],
      };
    }

    // Boundary conditions
    const bc: Json = {};
    for (const face of WALL_FACES) {
      const v = s.walls[face];
      bc[face] = v === 'Inflow' ? { Inflow: s.inflowParams ?? { velocity: [0, 0, 0], temperature: 293.15 } } : v;
    }
    config.boundary_conditions = bc;

    // Obstacle
    if (s.obstacle === 'sphere') {
      config.geometry = { type: 'sphere', center: [...s.obstacleCenter], radius: s.obstacleRadius };
    } else if (s.obstacle === 'cylinder') {
      config.geometry = {
        type: 'cylinder',
        center: [...s.obstacleCenter],
        radius: s.obstacleRadius,
        height: s.obstacleHeight,
        axis: s.obstacleAxis,
      };
    } else if (s.obstacle === 'box') {
      config.geometry = {
        type: 'box',
        min: [
          s.obstacleCenter[0] - s.obstacleBoxSize[0] / 2,
          s.obstacleCenter[1] - s.obstacleBoxSize[1] / 2,
          s.obstacleCenter[2] - s.obstacleBoxSize[2] / 2,
        ],
        max: [
          s.obstacleCenter[0] + s.obstacleBoxSize[0] / 2,
          s.obstacleCenter[1] + s.obstacleBoxSize[1] / 2,
          s.obstacleCenter[2] + s.obstacleBoxSize[2] / 2,
        ],
      };
    } else if (s.obstacle === 'stl' && s.geometryFile) {
      config.geometry_file = s.geometryFile;
    }

    if (s.maxTime != null) config.max_time = s.maxTime;
    if (s.speedOfSound != null) config.speed_of_sound = s.speedOfSound;
    if (s.cfl != null) config.cfl_number = s.cfl;
    if (s.initialTemperature != null) config.initial_temperature = s.initialTemperature;
    if (s.maxTimesteps != null) config.max_timesteps = s.maxTimesteps;

    return config;
  }

  /** The wall specs in the form (for container rendering) */
  wallSpec(): Record<WallFace, WallValue> {
    return { ...this.state.walls };
  }

  /** Estimated particle count for the current settings */
  estimateCount(): number {
    const s = this.state;
    const vol =
      s.domainSize[0] * s.fillFrac[0] *
      s.domainSize[1] * s.fillFrac[1] *
      s.domainSize[2] * s.fillFrac[2];
    return Math.round(vol / Math.pow(s.spacing, 3));
  }

  // -------------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------------

  private render(): void {
    const s = this.state;
    this.container.innerHTML = '';

    // --- Solver section ---
    const solver = this.section('Solver');
    solver.appendChild(
      this.selectRow('method', s.solver, [
        ['wcsph', 'WCSPH'],
        ['pcisph', 'PCISPH'],
      ], (v) => { this.state.solver = v as FormState['solver']; }),
    );
    solver.appendChild(
      this.selectRow('backend', s.backend, [
        ['auto', 'auto (GPU if available)'],
        ['gpu', 'GPU'],
        ['cpu', 'CPU'],
      ], (v) => { this.state.backend = v as FormState['backend']; }),
    );

    // Resolution: log-scale slider over particle spacing
    const spacingRow = this.sliderRow(
      'resolution',
      Math.log10(s.spacing),
      Math.log10(0.0008),
      Math.log10(0.01),
      0.01,
      (v) => {
        this.state.spacing = Math.pow(10, v);
        return `${(this.state.spacing * 1000).toFixed(2)}mm`;
      },
      `${(s.spacing * 1000).toFixed(2)}mm`,
    );
    solver.appendChild(spacingRow);
    const estimate = document.createElement('div');
    estimate.className = 'hint';
    estimate.id = 'particle-estimate';
    solver.appendChild(estimate);

    // --- Physics section ---
    const physics = this.section('Physics');
    physics.appendChild(
      this.selectRow('fluid', s.fluidType, [
        ['Water', 'water'],
        ['Air', 'air (CPU only)'],
        ['Mixed', 'water + air (unstable)'],
      ], (v) => { this.state.fluidType = v as FormState['fluidType']; }),
    );
    physics.appendChild(
      this.vectorRow('gravity', s.gravity, (vec) => { this.state.gravity = vec; }),
    );
    physics.appendChild(
      this.numberRow('viscosity', s.viscosity, 0.000001, (v) => { this.state.viscosity = v; }),
    );
    physics.appendChild(
      this.numberRow('max time s', s.maxTime ?? 0, 0.1, (v) => {
        this.state.maxTime = v > 0 ? v : null;
      }),
    );

    // --- Container section ---
    const container = this.section('Container');
    container.appendChild(
      this.vectorRow('size cm', [
        s.domainSize[0] * 100,
        s.domainSize[1] * 100,
        s.domainSize[2] * 100,
      ], (vec) => {
        this.state.domainSize = [vec[0] / 100, vec[1] / 100, vec[2] / 100];
      }),
    );

    const fillLabel = ['fill x', 'fill y', 'fill z'];
    for (let axis = 0; axis < 3; axis++) {
      container.appendChild(
        this.sliderRow(
          fillLabel[axis],
          s.fillFrac[axis],
          0.05,
          1.0,
          0.05,
          (v) => {
            this.state.fillFrac[axis] = v;
            return `${Math.round(v * 100)}%`;
          },
          `${Math.round(s.fillFrac[axis] * 100)}%`,
        ),
      );
    }

    const wallsTitle = document.createElement('div');
    wallsTitle.className = 'hint';
    wallsTitle.textContent = 'boundary conditions';
    container.appendChild(wallsTitle);

    const grid = document.createElement('div');
    grid.className = 'walls-grid';
    for (const face of WALL_FACES) {
      const cell = document.createElement('div');
      cell.className = 'wall-cell';
      const label = document.createElement('label');
      label.textContent = face.replace('_', ' ');
      cell.appendChild(label);

      const sel = document.createElement('select');
      const options: WallValue[] = ['Wall', 'Outflow', 'Periodic'];
      if (s.walls[face] === 'Inflow') options.push('Inflow');
      for (const opt of options) {
        const o = document.createElement('option');
        o.value = opt;
        o.textContent = opt.toLowerCase();
        if (s.walls[face] === opt) o.selected = true;
        sel.appendChild(o);
      }
      sel.addEventListener('change', () => {
        const v = sel.value as WallValue;
        this.state.walls[face] = v;
        // Periodic boundaries must be paired: mirror onto the opposite face
        if (v === 'Periodic') {
          const opposite = (face.endsWith('min')
            ? face.replace('min', 'max')
            : face.replace('max', 'min')) as WallFace;
          if (this.state.walls[opposite] !== 'Periodic') {
            this.state.walls[opposite] = 'Periodic';
            this.render();
          }
        }
        this.emitChange();
      });
      cell.appendChild(sel);
      grid.appendChild(cell);
    }
    container.appendChild(grid);

    // --- Obstacle section ---
    const obstacle = this.section('Obstacle');
    const kinds: [string, string][] = [
      ['none', 'none'],
      ['sphere', 'sphere'],
      ['box', 'box'],
      ['cylinder', 'cylinder'],
    ];
    if (s.obstacle === 'stl') kinds.push(['stl', `STL (${s.geometryFile?.split('/').pop()})`]);
    obstacle.appendChild(
      this.selectRow('shape', s.obstacle, kinds, (v) => {
        this.state.obstacle = v as ObstacleKind;
        // Default the obstacle into the middle of the domain
        const c: [number, number, number] = [
          this.state.domainMin[0] + this.state.domainSize[0] * 0.6,
          this.state.domainMin[1] + this.state.domainSize[1] * 0.5,
          this.state.domainMin[2] + this.state.domainSize[2] * 0.5,
        ];
        this.state.obstacleCenter = c;
        const minDim = Math.min(...this.state.domainSize);
        this.state.obstacleRadius = minDim * 0.15;
        this.state.obstacleHeight = this.state.domainSize[1];
        this.state.obstacleBoxSize = [minDim * 0.3, minDim * 0.3, minDim * 0.3];
        this.seatObstacle();
        this.render();
      }),
    );

    if (s.obstacle === 'sphere' || s.obstacle === 'cylinder' || s.obstacle === 'box') {
      obstacle.appendChild(
        this.vectorRow('center cm', [
          s.obstacleCenter[0] * 100,
          s.obstacleCenter[1] * 100,
          s.obstacleCenter[2] * 100,
        ], (vec) => {
          this.state.obstacleCenter = [vec[0] / 100, vec[1] / 100, vec[2] / 100];
        }),
      );
    }
    if (s.obstacle === 'sphere' || s.obstacle === 'cylinder') {
      obstacle.appendChild(
        this.numberRow('radius cm', s.obstacleRadius * 100, 0.05, (v) => {
          this.state.obstacleRadius = v / 100;
        }),
      );
    }
    if (s.obstacle === 'cylinder') {
      obstacle.appendChild(
        this.numberRow('height cm', s.obstacleHeight * 100, 0.05, (v) => {
          this.state.obstacleHeight = v / 100;
          this.seatObstacle();
          this.render();
        }),
      );
      obstacle.appendChild(
        this.selectRow('axis', s.obstacleAxis, [
          ['x', 'x'],
          ['y', 'y'],
          ['z', 'z'],
        ], (v) => {
          this.state.obstacleAxis = v as FormState['obstacleAxis'];
          this.seatObstacle();
          this.render();
        }),
      );
      const seatHint = document.createElement('div');
      seatHint.className = 'hint';
      seatHint.textContent = 'vertical pillars sit on the floor — raise center y to float';
      obstacle.appendChild(seatHint);
    }
    if (s.obstacle === 'box') {
      obstacle.appendChild(
        this.vectorRow('size cm', [
          s.obstacleBoxSize[0] * 100,
          s.obstacleBoxSize[1] * 100,
          s.obstacleBoxSize[2] * 100,
        ], (vec) => {
          this.state.obstacleBoxSize = [vec[0] / 100, vec[1] / 100, vec[2] / 100];
        }),
      );
    }

    this.updateEstimate();
  }

  /**
   * Rest the obstacle on the domain floor. A short pillar centered at
   * mid-height floats, and water flowing underneath looks like it passes
   * straight through the obstacle.
   */
  private seatObstacle(): void {
    const s = this.state;
    const floorY = s.domainMin[1];
    if (s.obstacle === 'cylinder' && s.obstacleAxis === 'y') {
      s.obstacleCenter[1] = floorY + s.obstacleHeight / 2;
    } else if (s.obstacle === 'box') {
      s.obstacleCenter[1] = floorY + s.obstacleBoxSize[1] / 2;
    }
  }

  private updateEstimate(): void {
    const el = this.container.querySelector('#particle-estimate');
    if (!el) return;
    const n = this.estimateCount();
    const pretty = n >= 1000 ? `${(n / 1000).toFixed(1)}K` : `${n}`;
    if (n > 150000) {
      el.innerHTML = `≈ <span class="warn">${pretty} particles — may be slow</span>`;
    } else {
      el.textContent = `≈ ${pretty} particles · coarser = faster`;
    }
  }

  // --- small DOM builders ---

  private section(title: string): HTMLElement {
    const sec = document.createElement('div');
    sec.className = 'section';
    const h = document.createElement('div');
    h.className = 'section-title';
    h.textContent = title;
    sec.appendChild(h);
    this.container.appendChild(sec);
    return sec;
  }

  private selectRow(
    label: string,
    value: string,
    options: [string, string][],
    onChange: (v: string) => void,
  ): HTMLElement {
    const row = document.createElement('div');
    row.className = 'field-row';
    const lab = document.createElement('label');
    lab.textContent = label;
    row.appendChild(lab);

    const sel = document.createElement('select');
    for (const [v, text] of options) {
      const o = document.createElement('option');
      o.value = v;
      o.textContent = text;
      if (v === value) o.selected = true;
      sel.appendChild(o);
    }
    sel.addEventListener('change', () => {
      onChange(sel.value);
      this.emitChange();
    });
    row.appendChild(sel);
    return row;
  }

  private sliderRow(
    label: string,
    value: number,
    min: number,
    max: number,
    step: number,
    onChange: (v: number) => string,
    initialDisplay: string,
  ): HTMLElement {
    const row = document.createElement('div');
    row.className = 'field-row';
    const lab = document.createElement('label');
    lab.textContent = label;
    row.appendChild(lab);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(value);
    row.appendChild(slider);

    const val = document.createElement('span');
    val.className = 'value';
    val.textContent = initialDisplay;
    row.appendChild(val);

    slider.addEventListener('input', () => {
      val.textContent = onChange(parseFloat(slider.value));
      this.emitChange();
    });
    return row;
  }

  private numberRow(
    label: string,
    value: number,
    step: number,
    onChange: (v: number) => void,
  ): HTMLElement {
    const row = document.createElement('div');
    row.className = 'field-row';
    const lab = document.createElement('label');
    lab.textContent = label;
    row.appendChild(lab);

    const input = document.createElement('input');
    input.type = 'number';
    input.step = String(step);
    input.value = String(Number(value.toPrecision(6)));
    input.addEventListener('change', () => {
      const v = parseFloat(input.value);
      if (Number.isFinite(v)) {
        onChange(v);
        this.emitChange();
      }
    });
    row.appendChild(input);
    return row;
  }

  private vectorRow(
    label: string,
    value: [number, number, number] | number[],
    onChange: (v: [number, number, number]) => void,
  ): HTMLElement {
    const row = document.createElement('div');
    row.className = 'field-row';
    const lab = document.createElement('label');
    lab.textContent = label;
    row.appendChild(lab);

    const triple = document.createElement('div');
    triple.className = 'triple';
    const inputs: HTMLInputElement[] = [];
    for (let i = 0; i < 3; i++) {
      const input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      input.value = String(Number(value[i].toPrecision(5)));
      inputs.push(input);
      triple.appendChild(input);
    }
    const handler = () => {
      const vec = inputs.map((inp) => parseFloat(inp.value)) as [number, number, number];
      if (vec.every(Number.isFinite)) {
        onChange(vec);
        this.emitChange();
      }
    };
    inputs.forEach((inp) => inp.addEventListener('change', handler));
    row.appendChild(triple);
    return row;
  }
}
