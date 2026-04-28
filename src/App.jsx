import React, { useState, useEffect, useMemo } from 'react';

// ============================================================================
// 1. MATHEMATICAL & PHYSICAL ENGINE (FIRST PRINCIPLES)
// ============================================================================

/**
 * Time-Varying Elastance E(t)
 * Models the active contraction and relaxation of the ventricular myocardium.
 * Derived from the normalized piecewise Double-Hill or Cosine formulations.
 */
const elastance = (t, hr, E_min, E_max) => {
  const tc = 60.0 / hr; // Duration of one cardiac cycle (R-R interval)
  
  // Systolic duration empirically scales with the cycle length (Bazett's roughly)
  // Physiological approximation: systole is roughly 30% of cycle at 75bpm
  const t_sys = 0.2 + 0.15 * (tc / 0.8); 
  const tm = t % tc; // Local time within the current cycle

  if (tm < t_sys) {
    // Systole: Active contraction (Rising cosine)
    const e_t = 0.5 * (1 - Math.cos(Math.PI * tm / t_sys));
    return E_min + (E_max - E_min) * e_t;
  } else if (tm < t_sys * 1.5) {
    // Isovolumic Relaxation (Falling cosine)
    const e_t = 0.5 * (1 + Math.cos(Math.PI * (tm - t_sys) / (t_sys * 0.5)));
    return E_min + (E_max - E_min) * e_t;
  } else {
    // Diastole: Passive filling
    return E_min;
  }
};

/**
 * State-Space Derivatives: dq/dt = A(t)q + B(t)u
 * Vector q = [q_lv, q_art, q_cap, q_ven]^T represents stressed volumes.
 */
const computeDerivatives = (t, q, params) => {
  const [qlv, qa, qc, qv] = q;
  const { hr, E_min, E_max, Ra, Rc, Rv, Ca, Cc, Cv, Rin, Rout } = params;

  // 1. Compute Time-Varying Elastance
  const E_t = elastance(t, hr, E_min, E_max);

  // 2. Compute Nodal Pressures (P = V/C)
  // For the ventricle, Elastance is the inverse of Compliance (E = 1/C)
  const Plv = qlv * E_t;
  const Pa = qa / Ca;
  const Pc = qc / Cc;
  const Pv = qv / Cv;

  // 3. Compute Flows (Currents) via Ideal Diodes with finite resistance
  // i = (P1 - P2) / R if P1 > P2 (valve open), else 0 (valve closed)
  const i_in = Pv > Plv ? (Pv - Plv) / Rin : 0.0;     // Mitral Valve Flow
  const i_out = Plv > Pa ? (Plv - Pa) / Rout : 0.0;   // Aortic Valve Flow

  // Systemic vascular flows (Ohm's Law for fluids: Q = dP / R)
  const i_ac = (Pa - Pc) / Ra;  // Arteries to Capillaries
  const i_cv = (Pc - Pv) / Rc;  // Capillaries to Veins

  // 4. Conservation of Mass (Kirchhoff's Current Law)
  // The rate of change of volume is the sum of flows in minus flows out
  return [
    i_in - i_out, // dq_lv/dt
    i_out - i_ac, // dq_art/dt
    i_ac - i_cv,  // dq_cap/dt
    i_cv - i_in   // dq_ven/dt
  ];
};

/**
 * 4th-Order Runge-Kutta Numerical Integrator
 * Crucial for stiff systems (like valve switching) to prevent numerical divergence.
 */
const rk4_step = (t, q, dt, params) => {
  const k1 = computeDerivatives(t, q, params);
  
  const q2 = q.map((qi, i) => qi + 0.5 * dt * k1[i]);
  const k2 = computeDerivatives(t + 0.5 * dt, q2, params);
  
  const q3 = q.map((qi, i) => qi + 0.5 * dt * k2[i]);
  const k3 = computeDerivatives(t + 0.5 * dt, q3, params);
  
  const q4 = q.map((qi, i) => qi + dt * k3[i]);
  const k4 = computeDerivatives(t + dt, q4, params);

  return q.map((qi, i) => qi + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
};

// ============================================================================
// 2. UI COMPONENTS & SVG PLOTTING
// ============================================================================

const ScientificPlot = ({ data, xKey, yKeys, colors, xLabel, yLabel, title, isPV = false }) => {
  if (!data || data.length === 0) return null;

  // Compute exact bounds
  let xMin = Math.min(...data.map(d => d[xKey]));
  let xMax = Math.max(...data.map(d => d[xKey]));
  let yMin = Number.MAX_VALUE;
  let yMax = Number.MIN_VALUE;

  data.forEach(d => {
    yKeys.forEach(k => {
      if (d[k] < yMin) yMin = d[k];
      if (d[k] > yMax) yMax = d[k];
    });
  });

  // Pad axes for visual breathing room
  const xPad = (xMax - xMin) * 0.1 || 1;
  const yPad = (yMax - yMin) * 0.1 || 1;
  xMin -= xPad; xMax += xPad;
  yMin = Math.max(0, yMin - yPad); yMax += yPad;

  const width = 600;
  const height = 350;
  const padding = { top: 40, right: 30, bottom: 50, left: 60 };

  const scaleX = (x) => padding.left + ((x - xMin) / (xMax - xMin)) * (width - padding.left - padding.right);
  const scaleY = (y) => height - padding.bottom - ((y - yMin) / (yMax - yMin)) * (height - padding.top - padding.bottom);

  // Generate SVG paths
  const paths = yKeys.map((key) => {
    const points = data.map(d => `${scaleX(d[xKey])},${scaleY(d[key])}`).join(' L ');
    // If it's a PV loop, close the path to ensure it looks perfectly unified
    return isPV ? `M ${points} Z` : `M ${points}`;
  });

  return (
    <div className="bg-white p-5 rounded-xl border border-neutral-200 shadow-sm flex flex-col h-full">
      <h3 className="font-bold text-neutral-800 mb-2 text-center">{title}</h3>
      <div className="flex-grow w-full relative">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full text-xs font-sans absolute inset-0">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map(tick => {
            const y = height - padding.bottom - tick * (height - padding.top - padding.bottom);
            const x = padding.left + tick * (width - padding.left - padding.right);
            return (
              <g key={`grid-${tick}`}>
                <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="#f3f4f6" strokeWidth="1" />
                <line x1={x} y1={padding.top} x2={x} y2={height - padding.bottom} stroke="#f3f4f6" strokeWidth="1" />
              </g>
            );
          })}

          {/* Axes */}
          <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="#9ca3af" strokeWidth="2" />
          <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="#9ca3af" strokeWidth="2" />
          
          {/* Y-axis labels */}
          {[0, 0.5, 1].map(tick => {
            const val = yMin + tick * (yMax - yMin);
            const y = scaleY(val);
            return <text key={`yl-${tick}`} x={padding.left - 10} y={y} textAnchor="end" alignmentBaseline="middle" fill="#6b7280">{Math.round(val)}</text>;
          })}

          {/* X-axis labels */}
          {[0, 0.5, 1].map(tick => {
            const val = xMin + tick * (xMax - xMin);
            const x = scaleX(val);
            return <text key={`xl-${tick}`} x={x} y={height - padding.bottom + 20} textAnchor="middle" fill="#6b7280">{isPV ? Math.round(val) : val.toFixed(1)}</text>;
          })}

          {/* Data Paths */}
          {paths.map((path, i) => (
            <path key={i} d={path} fill={isPV ? `${colors[i]}20` : 'none'} stroke={colors[i]} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
          ))}

          {/* Axis Titles */}
          <text x={width / 2} y={height - 10} textAnchor="middle" fill="#374151" className="font-semibold">{xLabel}</text>
          <text x={15} y={height / 2} textAnchor="middle" transform={`rotate(-90, 15, ${height/2})`} fill="#374151" className="font-semibold">{yLabel}</text>
          
          {/* Legend for Time Domain */}
          {!isPV && yKeys.length > 1 && (
            <g transform={`translate(${width - 100}, ${padding.top})`}>
              {yKeys.map((key, i) => (
                <g key={`legend-${key}`} transform={`translate(0, ${i * 15})`}>
                  <line x1="0" y1="0" x2="15" y2="0" stroke={colors[i]} strokeWidth="3" />
                  <text x="20" y="4" fill="#4b5563">{key.replace('P_', 'P ')}</text>
                </g>
              ))}
            </g>
          )}
        </svg>
      </div>
    </div>
  );
};

// ============================================================================
// 3. MAIN APPLICATION
// ============================================================================

export default function App() {
  // Physiological Baseline Parameters
  const [params, setParams] = useState({
    hr: 75,
    E_max: 2.5,      // Increased from 2.0 (Boosts contractility, raises EF to ~60%)
    E_min: 0.06,     // Diastolic stiffness
    V_stressed: 850, // Reduced from 1000mL (Fixes fluid overload, normalizes EDV & BP)
    Ra: 1.1,         // Arterial Resistance (Tuned for ~120/80 BP)
    Ca: 1.5,         // Arterial Compliance (Widens pulse pressure for healthier systolic peak)
    // Hidden standard physiological constants
    Rc: 0.2,         // Capillary resistance
    Cc: 3.0,         // Capillary compliance
    Rv: 0.05,        // Venous return resistance
    Cv: 50.0,        // Venous compliance
    Rin: 0.015,      // Mitral valve resistance
    Rout: 0.04,      // Aortic valve resistance
    V0_lv: 15.0      // Unstressed volume of LV (mL)
  });

  const [plotData, setPlotData] = useState([]);
  const [metrics, setMetrics] = useState({});

  useEffect(() => {
    const dt = 0.002; // 2ms integration step
    const stabilizationTime = 15.0; // Let system run for 15s to guarantee limit cycle
    const cycleDuration = 60.0 / params.hr; // Exact duration of one beat
    const totalSimTime = stabilizationTime + cycleDuration; // Stop exactly after 1 beat at steady state
    
    // Initial guess for stressed volume distribution
    let q = [
      params.V_stressed * 0.10, // Ventricle
      params.V_stressed * 0.20, // Arteries
      params.V_stressed * 0.10, // Capillaries
      params.V_stressed * 0.60  // Veins
    ];

    let results = [];
    let sysP = 0; let diaP = 1000;
    let minVol = 1000; let maxVol = 0;

    for (let t = 0; t <= totalSimTime; t += dt) {
      q = rk4_step(t, q, dt, params);

      // Only record data during the final cardiac cycle
      if (t >= stabilizationTime) {
        const E_t = elastance(t, params.hr, params.E_min, params.E_max);
        const P_lv = q[0] * E_t;
        const P_art = q[1] / params.Ca;
        const P_ven = q[3] / params.Cv;
        const V_lv_total = q[0] + params.V0_lv; // Add unstressed volume to get anatomical volume

        // Normalize time for the plot (start from 0)
        const plotTime = t - stabilizationTime;
        results.push({ t: plotTime, P_lv, P_art, P_ven, V_lv_total });

        // Metric Tracking
        if (P_art > sysP) sysP = P_art;
        if (P_art < diaP) diaP = P_art;
        if (V_lv_total > maxVol) maxVol = V_lv_total;
        if (V_lv_total < minVol) minVol = V_lv_total;
      }
    }

    setPlotData(results);
    setMetrics({
      bp: `${Math.round(sysP)}/${Math.round(diaP)}`,
      sv: Math.round(maxVol - minVol),
      edv: Math.round(maxVol),
      esv: Math.round(minVol),
      co: ((maxVol - minVol) * params.hr / 1000).toFixed(1),
      ef: Math.round(((maxVol - minVol) / maxVol) * 100)
    });
  }, [params]);

  const handleChange = (e) => {
    setParams({ ...params, [e.target.name]: parseFloat(e.target.value) });
  };

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 font-sans selection:bg-blue-200 p-4 md:p-8">
      
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header Section */}
        <header className="border-b border-neutral-300 pb-6">
          <h1 className="text-4xl font-black tracking-tight text-neutral-900 mb-2">Cardiovascular Hemodynamics Project</h1>
          <p className="text-xl text-neutral-600 font-medium">State-Space Analysis of a Lumped-Parameter Cardiovascular System</p>
          <div className="mt-4 inline-block bg-blue-50 text-blue-800 text-sm font-semibold px-3 py-1 rounded-full">
            Author: Vasig | IIT Madras | First-Principles Simulation
          </div>
        </header>

        {/* Top Grid: Theory & Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Theory / Equations (Left) */}
          <div className="lg:col-span-8 bg-white p-8 rounded-2xl shadow-sm border border-neutral-200">
            <h2 className="text-2xl font-bold mb-6 text-neutral-800 border-b pb-2">1. Mathematical Formulation</h2>
            
            <div className="prose prose-neutral max-w-none text-sm leading-relaxed">
              <p>
                This simulation employs a rigorous <strong>4th-order switched linear time-varying (LTV)</strong> model. By utilizing the electrical-fluid analogy, the circulatory system is mapped to a state-space framework where volume maps to charge (<span className="italic">q</span>), pressure maps to voltage (<span className="italic">V</span>), and flow maps to current (<span className="italic">I</span>).
              </p>

              <h4 className="font-semibold text-neutral-700 mt-6 mb-2">The State Vector</h4>
              <p>The state vector tracks the <em>stressed volume</em> (mL) across four physiological compartments:</p>
              <div className="bg-neutral-50 p-4 rounded-lg font-serif text-center text-lg border border-neutral-100">
                q(t) = [ q<sub>lv</sub>, q<sub>art</sub>, q<sub>cap</sub>, q<sub>ven</sub> ]<sup>T</sup>
              </div>

              <h4 className="font-semibold text-neutral-700 mt-6 mb-2">Time-Varying Elastance (The Pump)</h4>
              <p>The heart is modeled not as a static pressure source, but as a time-varying capacitor. Its inverse capacitance (Elastance) is defined by a normalized piecewise cosine function driving isovolumic contraction and relaxation.</p>
              <div className="bg-neutral-50 p-4 rounded-lg font-serif text-center border border-neutral-100 flex flex-col space-y-2">
                <span>P<sub>lv</sub>(t) = E(t) &middot; (V<sub>total</sub>(t) - V<sub>0</sub>)</span>
                <span className="text-sm text-neutral-500">where E(t) oscillates between E<sub>min</sub> (diastole) and E<sub>max</sub> (systole)</span>
              </div>

              <h4 className="font-semibold text-neutral-700 mt-6 mb-2">The Differential Equations</h4>
              <p>Applying Kirchhoff's Current Law ($\sum I = 0$), the mass conservation of blood flow is expressed as a system of Ordinary Differential Equations (ODEs). The valves are modeled as ideal diodes with finite physiological resistance (R<sub>in</sub>, R<sub>out</sub>), preventing instantaneous non-physical flows.</p>
              <div className="bg-neutral-50 p-4 rounded-lg font-serif grid grid-cols-1 md:grid-cols-2 gap-4 text-sm border border-neutral-100">
                <div>
                  <span className="font-bold text-blue-700">dq<sub>lv</sub>/dt</span> = i<sub>mitral</sub> - i<sub>aortic</sub><br/>
                  <span className="font-bold text-blue-700">dq<sub>art</sub>/dt</span> = i<sub>aortic</sub> - (P<sub>art</sub> - P<sub>cap</sub>)/R<sub>a</sub>
                </div>
                <div>
                  <span className="font-bold text-blue-700">dq<sub>cap</sub>/dt</span> = (P<sub>art</sub> - P<sub>cap</sub>)/R<sub>a</sub> - (P<sub>cap</sub> - P<sub>ven</sub>)/R<sub>c</sub><br/>
                  <span className="font-bold text-blue-700">dq<sub>ven</sub>/dt</span> = (P<sub>cap</sub> - P<sub>ven</sub>)/R<sub>c</sub> - i<sub>mitral</sub>
                </div>
              </div>
              <p className="mt-4 text-xs text-neutral-500 italic">
                * Note: The system is integrated using a custom 4th-Order Runge-Kutta (RK4) numerical solver with a 2ms timestep to ensure stability across the stiff valve-switching boundaries. The graphs below plot exactly one steady-state limit cycle.
              </p>
            </div>
          </div>

          {/* Controls (Right) */}
          <div className="lg:col-span-4 bg-white p-8 rounded-2xl shadow-sm border border-neutral-200">
            <h2 className="text-2xl font-bold mb-6 text-neutral-800 border-b pb-2">2. Parameters</h2>
            
            <div className="space-y-6">
              {[
                { name: 'hr', label: 'Heart Rate (bpm)', min: 40, max: 180, step: 1, desc: "Cardiac cycle frequency" },
                { name: 'E_max', label: 'Contractility (E_max)', min: 0.5, max: 4.0, step: 0.1, desc: "Peak systolic stiffness" },
                { name: 'E_min', label: 'Diastolic Relaxation (E_min)', min: 0.02, max: 0.20, step: 0.01, desc: "Passive filling compliance" },
                { name: 'Ra', label: 'Systemic Resistance (Ra)', min: 0.5, max: 3.0, step: 0.1, desc: "Arteriolar constriction" },
                { name: 'Ca', label: 'Arterial Compliance (Ca)', min: 0.5, max: 3.0, step: 0.1, desc: "Aortic elasticity (Windkessel)" }
              ].map(ctrl => (
                <div key={ctrl.name} className="group">
                  <div className="flex justify-between items-end mb-1">
                    <div>
                      <label className="text-sm font-bold text-neutral-800 block">{ctrl.label}</label>
                      <span className="text-xs text-neutral-500">{ctrl.desc}</span>
                    </div>
                    <span className="text-blue-600 font-black text-lg">{params[ctrl.name]}</span>
                  </div>
                  <input 
                    type="range" name={ctrl.name}
                    min={ctrl.min} max={ctrl.max} step={ctrl.step}
                    value={params[ctrl.name]} onChange={handleChange}
                    className="w-full accent-blue-600 h-2 bg-neutral-200 rounded-lg appearance-none cursor-pointer transition-all group-hover:bg-neutral-300"
                  />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Grid: Results & Visualization */}
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-neutral-800 pt-4">3. Hemodynamic Results (Steady State Limit Cycle)</h2>
          
          {/* Key Metrics Dashboard */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {[
              { label: "Blood Pressure", val: metrics.bp, unit: "mmHg", desc: "Systolic / Diastolic" },
              { label: "Stroke Volume", val: metrics.sv, unit: "mL", desc: "Volume ejected per beat" },
              { label: "Cardiac Output", val: metrics.co, unit: "L/min", desc: "Total systemic flow" },
              { label: "Ejection Fraction", val: metrics.ef, unit: "%", desc: "SV / EDV ratio" },
              { label: "End Diastolic Vol", val: metrics.edv, unit: "mL", desc: "Max LV filling volume" }
            ].map((m, i) => (
              <div key={i} className="bg-white border-l-4 border-blue-600 p-4 rounded-r-xl shadow-sm">
                <p className="text-xs font-bold text-neutral-500 uppercase tracking-wider">{m.label}</p>
                <div className="mt-1 flex items-baseline space-x-1">
                  <span className="text-2xl font-black text-neutral-900">{m.val}</span>
                  <span className="text-sm font-semibold text-neutral-500">{m.unit}</span>
                </div>
                <p className="text-[10px] text-neutral-400 mt-1">{m.desc}</p>
              </div>
            ))}
          </div>

          {/* Plots */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-[450px]">
            <ScientificPlot 
              data={plotData} 
              xKey="V_lv_total" 
              yKeys={['P_lv']} 
              colors={['#7e22ce']} 
              xLabel="Total Left Ventricular Volume (mL)" 
              yLabel="Left Ventricular Pressure (mmHg)" 
              title="Phase Space: Pressure-Volume (PV) Loop"
              isPV={true}
            />
            
            <ScientificPlot 
              data={plotData} 
              xKey="t" 
              yKeys={['P_lv', 'P_art', 'P_ven']} 
              colors={['#7e22ce', '#ef4444', '#3b82f6']} 
              xLabel="Time within Cycle (s)" 
              yLabel="Absolute Pressure (mmHg)" 
              title="Time-Domain: Wiggers Diagram Analog"
              isPV={false}
            />
          </div>
        </div>

      </div>
    </div>
  );
}