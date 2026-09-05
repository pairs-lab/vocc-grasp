% ============================================================
\section{Inputs to fusion}
% ============================================================

For every $(i,j)\in\cP_{\mathrm{cand}}$, the geometric branch returns
$(u^{cv}_{ij},p^{cv}_{ij},r_{ij},\delta^{cv}_{ij})$ from
Section~\ref{sec:uoais-ft}.  The VLM branch independently returns the raw score
$p^{vlm}_{ij}$ from~\eqref{eq:pvlm}; VLM calibration is performed in
Definition~\ref{def:cal}.

Define the prior-centered source evidences
\begin{equation}
\lambda^{cv}_{ij}=u^{cv}_{ij}-\logit(\pi_0),
\qquad
\lambda^{vlm}_{ij}=\logit(p^{vlm,cal}_{ij})-\logit(\pi_0).
\label{eq:source-evidence}
\end{equation}
If a source score were a calibrated posterior under the same prior, its centered logit
would equal a log-likelihood ratio.  In the implemented system this interpretation is
valid for the calibrated VLM only to the extent supported by held-out calibration; the
geometric term remains a signed confidence contribution.

\textbf{Output of this section:}
$\{\lambda^{cv}_{ij},\lambda^{vlm}_{ij},r_{ij},
\delta^{cv}_{ij},\delta^{vlm}_{ij}\}_{(i,j)\in\cP_{\mathrm{cand}}}$.

\section{Bayes-motivated per-edge fusion}
\label{sec:fusion}
% ============================================================

\textbf{Goal:} combine independently generated geometric and VLM evidence into one
probability $p_{ij}$ for each candidate edge.

\subsection{Ideal Bayes reference model}

Bayes' rule gives
\begin{equation}
\Prob(e_{ij}=1\mid\phi^{cv}_{ij},R^{vlm}_{ij})
\propto
\Prob(\phi^{cv}_{ij},R^{vlm}_{ij}\mid e_{ij}=1)\Prob(e_{ij}=1).
\label{eq:bayes-edge}
\end{equation}

\begin{assumption}[H1: cross-modal conditional independence]
\label{ass:h1}
On the frozen candidate-pair population, the geometric observation and VLM response are
modeled as conditionally independent given the true edge:
\begin{equation}
\Prob(\phi^{cv}_{ij},R^{vlm}_{ij}\mid e_{ij})
=
\Prob(\phi^{cv}_{ij}\mid e_{ij})
\Prob(R^{vlm}_{ij}\mid e_{ij}).
\label{eq:h1}
\end{equation}
\end{assumption}

H1 is an idealization, not an empirical identity: both branches observe the same physical
scene and the 3D network also consumes RGB.  Its purpose is to motivate additive log
evidence.  No geometric hint is passed to the VLM, which avoids the stronger and clearly
invalid dependence created by prompt-side geometric conditioning.

Under H1, and only when each source output is a calibrated posterior under the same prior
$\pi_0$, posterior log-odds take the ideal form
\begin{equation}
\logit p_{ij}
=
\logit\pi_0
+
\mathrm{LR}^{cv}_{ij}
+
\mathrm{LR}^{vlm}_{ij},
\qquad
\mathrm{LR}^{m}_{ij}=\logit p^{m}_{ij}-\logit\pi_0.
\label{eq:ideal-bayes}
\end{equation}
Because $p^{cv}_{ij}$ is not claimed to be separately calibrated, the implemented model
below is a learned logarithmic opinion pool with this Bayes structure, not a Bayes-exact
posterior.

\subsection{Stage-A calibration of VLM magnitude}

\begin{definition}[Scene-adaptive positive-slope Platt map]
\label{def:cal}
Let
\begin{equation}
p^{\epsilon}_{ij}
=
\min\{\max(p^{vlm}_{ij},\epsilon),1-\epsilon\},
\qquad \epsilon=10^{-6}.
\end{equation}
For scene $s$, with $N_{\mathrm{cand},s}=|\cP_{\mathrm{cand},s}|$, define
\begin{equation}
\phi_s
=
\frac{\log(1+N_{\mathrm{cand},s})-\mu_{\phi}}{\sigma_{\phi}},
\quad
a_s=\exp\!\left(\operatorname{clip}(\alpha_0+\alpha_N\phi_s,-5,5)\right)>0,
\quad
c_s=c_0+c_N\phi_s.
\end{equation}
The calibrated VLM probability is
\begin{equation}
\boxed{
p^{vlm,cal}_{ij}
=
\sigma\!\left[a_s\logit(p^{\epsilon}_{ij})+c_s\right].
}
\label{eq:vlm-cal}
\end{equation}
All calibration parameters and $(\mu_{\phi},\sigma_{\phi})$ are estimated without test
data and frozen before fusion training.
\end{definition}

For a fixed scene, $a_s>0$ makes the transformation monotone, so the within-scene VLM
ranking is preserved except for ties induced by clipping.  Setting $\alpha_N=c_N=0$
recovers global Platt scaling; additionally setting $c_0=0$ gives temperature-type
scaling.  The adaptive form is retained only when it improves pre-registered held-out
calibration criteria over the global map.

\subsection{Implemented reliability-weighted logit pool}

Using~\eqref{eq:source-evidence}, define
\begin{tcolorbox}[colback=blue!5,colframe=blue!50!black]
\begin{equation}
\boxed{
\logit(p_{ij})
=
\gamma
+
\delta^{vlm}_{ij}\beta_{vlm}\lambda^{vlm}_{ij}
+
\delta^{cv}_{ij}\beta_{cv}\lambda^{cv}_{ij}
}
\label{eq:fusion}
\end{equation}
\end{tcolorbox}
where $\beta_{vlm},\beta_{cv}\ge0$ and $\gamma$ are learned with the Stage-A map
frozen.  The signed depth evidence is already reliability-gated inside $u^{cv}_{ij}$ via
$d_{ij}=r_{ij}\tanh(\Delta z_{ij}/\sigma_z)$; therefore missing depth neutralizes only
the depth contribution and does not suppress valid mask-based evidence.

The availability indicators mean source failure, not source disagreement:
\begin{equation}
\delta^{vlm}_{ij}=\indic[\text{a valid VLM score is returned}],
\qquad
\delta^{cv}_{ij}=\indic[\text{the geometric feature vector is computable}].
\end{equation}
In particular, a distant pair with $o_{ij}=c_{ij}=0$ still has
$\delta^{cv}_{ij}=1$ and contributes negative geometric evidence.  Setting
$\delta^{cv}_{ij}=0$ for ``no overlap'' would incorrectly convert evidence of absence
into absence of evidence.

\subsection{Mechanism assignment for the main failure modes}

\begin{center}
\begin{tabular}{@{}p{4.1cm}p{5.8cm}p{5.4cm}@{}}
\toprule
\textbf{Failure mode} & \textbf{Mathematical symptom} & \textbf{Assigned mechanism} \\
\midrule
VLM magnitude error & predicted confidence differs from empirical frequency & Stage-A calibration \\
Local spatial hallucination & high edge score for a distant pair with weak hidden/clearance support & $o_{ij},c_{ij}$ in the UOAIS-FT geometric logit and optional frozen screening \\
Depth-direction inconsistency & edge $i\to j$ receives support although reliable depth places $j$ behind $i$ & signed term $d_{ij}$ in $u^{cv}_{ij}$ \\
Global structural inconsistency & individually plausible edges form a directed cycle & DAG support $\indic[\bfe\in\cD]$ \\
Off-chain object selection & an object not connected to the target chain is proposed as a blocker & reachability inside $\Free(X,\bfe)$ \\
\bottomrule
\end{tabular}
\end{center}

These mechanisms are deliberately not interchangeable.  DAG conditioning does not repair
a wrong depth order; the reachability operator does not remove a hallucinated edge that
itself makes a distant object reachable; and point calibration does not eliminate cyclic
joint mass.

\subsection{Training and source separation}
\label{sec:train}

\begin{description}[leftmargin=1.9em,topsep=2pt,itemsep=3pt]
\item[Stage 0a: UOAIS-FT.] Fine-tune the amodal segmentation front end on the frozen 2K
synthetic UNOBench subset using~\eqref{eq:uoais-ft}.
\item[Stage 0b: geometric score.] Select
$(b^{cv},\kappa_o,\kappa_c,\kappa_z,o_0,c_0,\rho,\sigma_z)$ on training/validation data
and freeze them.  The test set is never used to choose the geometric score.
\item[Stage A: VLM calibration.] Fit Definition~\ref{def:cal} by NLL and freeze it.
\item[Stage B: fusion.] Learn $(\gamma,\beta_{vlm},\beta_{cv})$ by regularized binary
cross-entropy on edge labels with all preceding stages frozen.
\end{description}

The downstream graph theory requires only that the final $p_{ij}$ lie in $(0,1)$.
Consequently, the DAG, Top-$K$, certificate, and decision results are insulated from the
specific valid implementation of the upstream pool.

% =====================================================================
\section{Joint structural model with the DAG constraint}
\label{sec:posterior}
% ============================================================

\textbf{Goal:} transform fused edge probabilities into a distribution over globally
consistent precedence structures.

\subsection{Local geometric errors versus global structural errors}
\label{sec:two-kinds}

A local edge may be wrong because its spatial support is absent, its depth direction is
inconsistent, or its confidence magnitude is incorrect.  Those errors are handled before
the joint graph model.  A structural error is different: several individually plausible
edges may jointly violate the strict precedence semantics.  The canonical example is a
directed cycle.  If $i\to j$ means that $j$ must be removed before $i$, no physical
removal order can satisfy a directed cycle.

Define the target-chain relation
\begin{equation}
\mathcal R_X(\bfe)=\Reach(X,\bfe)\setminus\{X\}.
\label{eq:target-reach}
\end{equation}
A false local edge may create false reachability.  Therefore, reachability is not used as
a global support constraint: doing so would merely accept the hallucinated path.  Instead,
local geometry suppresses unsupported edges, while $\Reach$ is used downstream to ensure
that disconnected objects are not counted as target blockers.

\begin{remark}[Decoded feasibility does not imply belief feasibility]
A thresholded or MAP graph can be acyclic while the edge-factorized model still assigns
substantial mass to cyclic configurations.  Decoded cycle rate and infeasible probability
mass therefore answer different questions.
\end{remark}

\subsection{Independent-edge reference model}

\begin{assumption}[H2: independent-edge product measure]
\label{ass:h2}
Before imposing global structure, the joint belief is modeled as the maximum-entropy
product distribution having the fused marginals $p_{ij}$:
\begin{equation}
P_{\otimes}(\bfe\mid\Omega)
=
\prod_{(i,j)\in\cP_{\mathrm{cand}}}
p_{ij}^{e_{ij}}(1-p_{ij})^{1-e_{ij}}
=:
W(\bfe).
\label{eq:prodmeasure}
\end{equation}
\end{assumption}

H2 is a declared modeling choice, not the native joint belief of either perception model.
The product measure is normalized on the full binary configuration space.

\subsection{Conditioning on acyclic support}

The feasible set is
\begin{equation}
\cD
=
\{\bfe\in\{0,1\}^{|\cP_{\mathrm{cand}}|}:G(\bfe)\text{ is acyclic}\}.
\label{eq:feasible-dag}
\end{equation}
The DAG-conditioned distribution is
\begin{tcolorbox}[colback=blue!5,colframe=blue!50!black]
\begin{equation}
\boxed{
\PD(\bfe\mid\Omega)
=
\frac{\indic[\bfe\in\cD]W(\bfe)}{Z},
\qquad
Z=\sum_{\bfe\in\cD}W(\bfe).
}
\label{eq:posterior}
\end{equation}
\end{tcolorbox}
Equivalently,
\begin{align}
\ell(\bfe)
&=
\sum_{(i,j)\in\cP_{\mathrm{cand}}}
\left[e_{ij}\log p_{ij}+(1-e_{ij})\log(1-p_{ij})\right],
\label{eq:log-weight}\\
W(\bfe)&=\exp\ell(\bfe).
\label{eq:weight}
\end{align}
This is a globally coupled factor distribution with one acyclicity factor, not a product
distribution after conditioning.

\begin{definition}[Model-implied mass outside DAG support]
\label{def:mu}
\begin{equation}
\mu
=
1-Z
=
P_{\otimes}(\bfe\notin\cD\mid\Omega).
\label{eq:mu}
\end{equation}
$\mu$ is a diagnostic of the declared product model.  It is not a probability of grasp
failure, a physical-safety quantity, or the inaccessible native joint belief of the VLM.
\end{definition}

No global reachability constraint is added to $\cD$.  Target relevance belongs in the
event defining $\Free(X,\bfe)$, while acyclicity belongs in the support.  This separation
prevents the structural layer from being credited for correcting local geometric
hallucinations that it cannot identify.

\section{Bayesian decision theory}
\label{sec:decision}
% ============================================================

\subsection{Graph-valid action marginals}

The losses below assess validity under the inferred obstruction graph.  They do not model
robot kinematics, grasp execution, or physical safety.  For direct target grasp,
\begin{equation}
L(\bfe,X)=\indic[\deg_X^+(\bfe)>0],
\qquad
q_X=P(\deg_X^+(\bfe)=0\mid\Omega,\bfe\in\cD),
\qquad
\rho(X)=1-q_X.
\label{eq:target-risk}
\end{equation}
For a non-target object $o$,
\begin{equation}
L(\bfe,o)=\indic[o\notin\Free(X,\bfe)],
\qquad
q_o=P(o\in\Free(X,\bfe)\mid\Omega,\bfe\in\cD),
\qquad
\rho(o)=1-q_o.
\label{eq:blocker-risk}
\end{equation}
Define
\begin{equation}
s_a=
\begin{cases}
q_X,&a=X,\\
q_o,&a=o\in V_t\setminus\{X\}.
\end{cases}
\label{eq:action-score}
\end{equation}
Under unit 0--1 loss, the Bayes action without defer is
\begin{equation}
a^*=\arg\max_{a\in V_t}s_a.
\label{eq:ostar-true}
\end{equation}

\subsection{Defer as a separate decision cost}

Extend the action set with $d=\textsf{defer}$ and assign constant cost
$L(\bfe,d)=\lambda_{d}\in(0,1)$.  Then
\begin{equation}
\tau_{\mathrm{act}}=1-\lambda_d
\end{equation}
and the Bayes rule is
\begin{equation}
a^*=
\begin{cases}
\arg\max_{a\in V_t}s_a,&\max_a s_a>\tau_{\mathrm{act}},\\[2pt]
\textsf{defer},&\max_a s_a\le\tau_{\mathrm{act}}.
\end{cases}
\label{eq:defer}
\end{equation}
At equality, acting and deferring have equal posterior risk; the implementation breaks the
tie in favor of defer.  Failure of the Top-$K$ certificate does not itself trigger defer:
certification concerns approximation error, whereas defer concerns posterior action risk.

\subsection{Free-set identification uses a different loss}

For a predicted set $\widehat F\subseteq V_t\setminus\{X\}$, use
\begin{equation}
L(\bfe,\widehat F)
=
c_{FP}\sum_{o\in\widehat F}\indic[o\notin\Free(X,\bfe)]
+
c_{FN}\sum_{o\notin\widehat F}\indic[o\in\Free(X,\bfe)].
\label{eq:setloss}
\end{equation}
Its posterior risk decomposes objectwise:
\begin{equation}
\rho(\widehat F)
=
c_{FP}\sum_{o\in\widehat F}(1-q_o)
+
c_{FN}\sum_{o\notin\widehat F}q_o.
\label{eq:setrisk}
\end{equation}
Therefore the Bayes-optimal set is
\begin{equation}
\widehat F^*
=
\{o:q_o>\tau_{\mathrm{set}}\},
\qquad
\tau_{\mathrm{set}}
=
\frac{c_{FP}}{c_{FP}+c_{FN}}.
\label{eq:setrule}
\end{equation}
The action threshold $\tau_{\mathrm{act}}$ and free-set threshold
$\tau_{\mathrm{set}}$ are conceptually distinct and are frozen separately.  They may be
tied as an implementation convention, but the theory does not require equality.

\begin{remark}[Distinct origins of the two thresholds]
The two thresholds arise from different decision problems:
\[
\tau_{\mathrm{act}}=1-\lambda_d
\]
is induced by the cost $\lambda_d$ of deferring relative to the unit
loss of executing an invalid immediate action, whereas
\[
\tau_{\mathrm{set}}
=
\frac{c_{FP}}{c_{FP}+c_{FN}}
\]
is induced by the asymmetric false-positive and false-negative costs
of free-set identification. Therefore, the theory does not require
$\tau_{\mathrm{act}}=\tau_{\mathrm{set}}$. They coincide only when
\[
1-\lambda_d
=
\frac{c_{FP}}{c_{FP}+c_{FN}},
\]
or equivalently,
\[
\lambda_d
=
\frac{c_{FN}}{c_{FP}+c_{FN}}.
\]
\end{remark}

\subsection{Top-$K$ policy interface}

At runtime, replace $q_X,q_o$ by their Top-$K$ estimates:
\begin{equation}
\widehat s_a^{(K)}=
\begin{cases}
\widehat q_X^{(K)},&a=X,\\
\widehat q_o^{(K)},&a=o.
\end{cases}
\label{eq:action-score-topk}
\end{equation}
The deployed outputs are
\begin{equation}
\widehat a^{(K)}=
\begin{cases}
\arg\max_{a\in V_t}\widehat s_a^{(K)},
&\max_a\widehat s_a^{(K)}>\tau_{\mathrm{act}},\\[2pt]
\textsf{defer},&\text{otherwise},
\end{cases}
\label{eq:policy}
\end{equation}
and
\begin{equation}
\widehat F^{(K)}
=
\{o\in V_t\setminus\{X\}:\widehat q_o^{(K)}>\tau_{\mathrm{set}}\}.
\label{eq:freeset-pred}
\end{equation}
Action selection and set identification are two decisions built from the same marginals;
using only the argmax to evaluate a multi-object free set would create artificial false
negatives.

\begin{definition}[Decision margin region]
\label{def:margin}
For a uniform marginal error bound $\epsilon_K$, define
\begin{equation}
\partial_{\epsilon_K}
=
\{o:|\widehat q_o^{(K)}-\tau_{\mathrm{set}}|\le\epsilon_K\}.
\end{equation}
Outside this band, Top-$K$ and exact free-set membership coincide.
\end{definition}

\subsection*{Worked example generated by exact enumeration}

This example is generated programmatically under the acyclic feasible set
$\cD$. The final version reports the candidate-edge probabilities,
the exact values of $Z$, $\bar Z$, and $\mu$, the direct-target marginal
$q_X$, all non-target free marginals $q_o$, the Single-MAP action,
the Top-$K$ action, the actual marginal approximation error,
and the certificate value $\epsilon_K$.

No numerical value is inserted into the paper until it has been reproduced
by the exact-enumeration harness.

% ============================================================
\section{Marginalization and Top-$K$ approximation}
\label{sec:marginal}
% ============================================================

\subsection{Defining $q_o$ via marginalization}

\begin{theorem}[Marginalization of Free-set indicator]
Let $Y_o(\bfe) := \indic[o \in \Free(X, \bfe)]$. Then:
\begin{equation}
q_o = \Prob(o \in \Free(X, \bfe) \mid \Omega) = \sum_{\bfe \in \cD} \indic[o \in \Free(X, \bfe)] \cdot \PD(\bfe\mid\Omega)
\label{eq:qo-marginal}
\end{equation}
\end{theorem}

\begin{proof}
Since $Y_o \in \{0, 1\}$, we have $\E[Y_o \mid \Omega] = \Prob(Y_o = 1 \mid \Omega) = q_o$. By the law of the unconscious statistician:
$$\E[Y_o \mid \Omega] = \sum_{\bfe} Y_o(\bfe) \cdot \PD(\bfe\mid\Omega).$$
Since $\PD(\bfe\mid\Omega) = 0$ for $\bfe \notin \cD$, the sum restricts to $\cD$. \qedhere
\end{proof}

\paragraph{Direct-target marginal.}
The graph-level direct-target marginal is
\begin{equation}
q_X
:=
\Prob\!\left(
\deg_X^+(\bfe)=0
\mid\Omega
\right).
\label{eq:qX}
\end{equation}
It is the posterior probability that the inferred obstruction graph contains no outgoing blocker from $X$; it is not a physical-grasp-success probability.

\subsection{Substitute posterior}

From \eqref{eq:posterior}, $\PD(\bfe\mid\Omega) = W(\bfe)/Z$ for $\bfe \in \cD$:
\begin{equation}
q_o = \frac{1}{Z} \sum_{\bfe \in \cD} W(\bfe) \cdot \indic[o \in \Free(X, \bfe)]
\label{eq:qo-explicit}
\end{equation}

\subsection{Computational problem}

$|\cD|$ is super-exponential ($\sim 10^9$ for $n=7$). Computing \eqref{eq:qo-explicit} exactly is intractable. An approximation is needed.

\subsection{Top-$K$ approximation}

Sort $\cD$ by $W$ in decreasing order: $W^{(1)} \geq W^{(2)} \geq \ldots \geq W^{(|\cD|)}$ with corresponding configurations $\hat\bfe^{(1)}, \hat\bfe^{(2)}, \ldots$.

\begin{definition}[Top-$K$ estimator]
\begin{equation}
\hat q_o^{(K)} := \frac{\sum_{k=1}^K \indic[o \in F^{(k)}] \cdot W^{(k)}}{\sum_{k=1}^K W^{(k)}}, F^{(k)} = \Free(X, \hat\bfe^{(k)})
\label{eq:qo-topk}
\end{equation}
\end{definition}

The corresponding Top-$K$ estimate for the direct-target is:
\begin{equation}
\widehat q_X^{(K)} =
\frac{\sum_{k=1}^{K}\indic[\deg_X^+(\widehat\bfe^{(k)})=0]\,r^{(k)}}
     {\sum_{k=1}^{K}r^{(k)}}.
\label{eq:qX-topk}
\end{equation}

\begin{remark}[Target and blockers cannot both be confidently free]
Let $Y_X(\bfe) := \indic[\deg_X^+(\bfe)=0]$.
Pointwise, $\deg_X^+(\bfe)=0$ implies $\Reach(X,\bfe)=\{X\}$, so no blocker is free; hence $Y_o(\bfe)\le 1-Y_X(\bfe)$ and $\widehat q_o^{(K)}\le1-\widehat q_X^{(K)}$ exactly. In particular, when $\tau_{\mathrm{set}}\ge\tfrac12$, a confidently free target ($\widehat q_X^{(K)}>\tau_{\mathrm{set}}$) forces $\widehat F^{(K)}=\varnothing$.
\end{remark}

\subsection{Error bound (truncation)}

\begin{lemma}[Truncation error bound]
\label{lem:error}
Let $Z_K = \sum_{k \leq K} W^{(k)}$ and $\text{tail}_K = \sum_{k > K} W^{(k)} = Z - Z_K$. Then
\begin{equation}
\big| q_o - \hat q_o^{(K)} \big| \leq \frac{\text{tail}_K}{Z_K + \text{tail}_K}.
\label{eq:err-bound}
\end{equation}
\end{lemma}

\begin{proof}
Let $A_K = \sum_{k \leq K} \indic[o \in F^{(k)}] W^{(k)}$ and
$B_K = \sum_{k > K} \indic[o \in F^{(k)}] W^{(k)}$. Then
$q_o = (A_K + B_K)/(Z_K + \text{tail}_K)$ and $\hat q_o^{(K)} = A_K/Z_K$, so
\[
q_o - \hat q_o^{(K)} = \frac{Z_K B_K - A_K \, \text{tail}_K}{Z_K(Z_K + \text{tail}_K)}.
\]
Since $0 \leq A_K \leq Z_K$ and $0 \leq B_K \leq \text{tail}_K$, the numerator has absolute value
$\leq Z_K \cdot \text{tail}_K$, giving $|q_o - \hat q_o^{(K)}| \leq \text{tail}_K/(Z_K + \text{tail}_K)$.
\end{proof}

\begin{corollary}
When $\text{tail}_K / Z \to 0$ (concentrated posterior), $\hat q_o^{(K)} \to q_o$. This lemma is
an intermediate step for the tail-model-free certificate (Theorem~\ref{thm:cert}).
\end{corollary}

% ============================================================
\section{Top-$K$ MAP via Integer Linear Programming}
\label{sec:topk-ilp}
% ============================================================

\textbf{Goal:} find the top-$K$ DAGs $\hat\bfe^{(k)}$ deterministically.

\subsection{Reformulation as an ILP}

We seek:
\begin{equation}
\hat\bfe^{(k)} = \arg\max_{\bfe \in \cD_k} \PD(\bfe\mid\Omega)
\quad\text{with}\quad \cD_k = \cD \setminus \{\hat\bfe^{(1)}, \ldots, \hat\bfe^{(k-1)}\}
\label{eq:kth-map}
\end{equation}

\begin{lemma}[Equivalence to a linear objective]
\eqref{eq:kth-map} is equivalent to:
\begin{equation}
\hat\bfe^{(k)} = \arg\max_{\bfe \in \cD_k} \sum_{(i,j) \in \cP_{\mathrm{cand}}} w_{ij} \cdot e_{ij}
\label{eq:ilp-obj}
\end{equation}
with
\begin{equation}
w_{ij} = \logit(p_{ij}) = \log\frac{p_{ij}}{1-p_{ij}}
\label{eq:wij}
\end{equation}
\end{lemma}

\begin{proof}
Take the log of \eqref{eq:posterior} for $\bfe \in \cD$:
\begin{align*}
\log \PD(\bfe\mid\Omega) &= \sum_{ij}\big[e_{ij}\log p_{ij} + (1-e_{ij})\log(1-p_{ij})\big] - \log Z \\
&= \sum_{ij} e_{ij}\big[\log p_{ij} - \log(1-p_{ij})\big] + \underbrace{\sum_{ij}\log(1-p_{ij}) - \log Z}_{\text{constant}} \\
&= \sum_{ij} e_{ij} \cdot w_{ij} + \text{const}.
\end{align*}
The argmax does not depend on the constant. \qedhere
\end{proof}

\subsection{ILP constraints}
The decision variables satisfy $z_{ij}\in\{0,1\}$ for every $(i,j)\in\cP_{\mathrm{cand}}$.


\paragraph{(a) Acyclicity (DAG).} For each cycle $C \subseteq \cP_{\mathrm{cand}}$:
\begin{equation}
\sum_{(a,b) \in C} z_{ab} \leq |C| - 1
\label{eq:dag-cut}
\end{equation}
Implemented with lazy constraints (detect cycle, add cut, resolve).

\paragraph{(b) No-good constraints} for each previously returned solution $\hat\bfe^{(h)}$, $h<k$:
\begin{equation}
\sum_{(i,j): \hat e^{(h)}_{ij}=1}(1-z_{ij}) + \sum_{(i,j): \hat e^{(h)}_{ij}=0}z_{ij} \geq 1
\label{eq:nogood}
\end{equation}
Force $\bfz \neq \hat\bfe^{(j)}$ in at least one position.

\begin{remark}[ILP solution]
The finite binary ILP is solved by branch-and-bound, with directed-cycle
constraints added lazily whenever an integer incumbent contains a cycle.
Sequential no-good cuts exclude previously returned solutions.
\end{remark}

\subsection{Adaptive stopping for top-$K$}
\label{sec:adaptive}

Iterate \eqref{eq:ilp-obj} for $k = 1, 2, \ldots$; the \textbf{output} is the set
$\{(\hat\bfe^{(k)}, \ell^{(k)}, F^{(k)})\}_{k=1}^K$. A fixed $K$ does not reflect
the actual complexity of each scene: simple scenes converge after a few configurations, complex scenes
need more. The \emph{ideal} criterion is to stop when the unexplored probability mass is small enough:
\begin{equation}
K^* = \min\Bigl\{k : \tfrac{Z_k}{Z} > 1-\epsilon\Bigr\}.
\label{eq:k*}
\end{equation}
But \eqref{eq:k*} is not directly computable because $Z=\sum_{\bfe\in\cD}W(\bfe)$ is the partition
function --- the very intractable quantity that top-$K$ is designed to avoid. We replace it with an
\emph{tail-model-free certificate} using only computed quantities (Theorem~\ref{thm:cert} below),
directly bounding the error of Lemma~\ref{lem:error},
$\bigl|q_o-\hat q_o^{(K)}\bigr|\le \text{tail}_K/(Z_K+\text{tail}_K)$, without knowing $Z$.

\medskip
\noindent\textbf{Notation.} The solver returns configurations in decreasing order $W_{(1)}\ge W_{(2)}\ge\cdots$;
let $Z_K=\sum_{k\le K}W_{(k)}$ (computed), $\text{tail}_K=\sum_{k>K}W_{(k)}$ (not computable),
$r^{(k)}=e^{\ell^{(k)}-\ell_{\max}}$.
 
\begin{proposition}[Computable upper bound on acyclic mass]\label{prop:zbar}
Let $Z=P_\otimes(\cD)$ with $\cD=\{\bfe:G(\bfe)\text{ is acyclic}\}$. Define
\begin{equation}
\bar Z := \prod_{i<j}\left(1-p_{ij}p_{ji}\right),
\label{eq:zbar}
\end{equation}
For the purpose of the unordered-pair product only, set $p_{ij}=0$ when $(i,j)\notin\cP_{\mathrm{cand}}$. Then $Z\le\bar Z$.
\end{proposition}
\begin{proof}
For each unordered pair $\{i,j\}$ let $C_{ij}:=\{e_{ij}=e_{ji}=1\}$. The events $C_{ij}$ depend on disjoint directed-edge coordinate pairs and are independent under $P_\otimes$. Every acyclic configuration avoids every $C_{ij}$, so
\[Z = P_\otimes(\cD) \le P_\otimes\!\Big(\bigcap_{i<j}C_{ij}^{c}\Big)= \prod_{i<j}(1-p_{ij}p_{ji}) = \bar Z. \qedhere\]
\end{proof}

\begin{theorem}[Tail-model-free stopping certificate]
\label{thm:cert}
Under \eqref{eq:posterior}, let $\hat\bfe^{(1)},\dots,\hat\bfe^{(K)}$ be \emph{any}
$K$ distinct elements of $\cD$, $Z_K=\sum_{k\le K}W(\hat\bfe^{(k)})$, and
$\hat q_o^{(K)}=Z_K^{-1}\sum_{k\le K}\indic[o\in\Free(X,\hat\bfe^{(k)})]W(\hat\bfe^{(k)})$.
With $\bar Z$ from Prop.~\ref{prop:zbar}: (i) $Z_K\le Z\le\bar Z$; (ii) for every $o$,
$|q_o-\hat q_o^{(K)}|\le\epsilon_K:=(\bar Z-Z_K)/\bar Z$; (iii) $\bar Z$ is computed once in $O(n^2)$ worst-case time; after that, $\epsilon_K$ is updated in $O(1)$ time given $Z_K$ and is non-increasing in $K$.
\end{theorem}
\begin{proof}
(i) $Z_K\le Z$ (distinct feasible, $W>0$); $Z\le\bar Z$ is Prop.~\ref{prop:zbar}.
(ii) With $\mathrm{tail}_K=Z-Z_K$, $A_K=\sum_{k\le K}\indic[o\in F^{(k)}]W(\hat\bfe^{(k)})$,
$B_K=\sum_{\bfe\in\cD\setminus\{\widehat\bfe^{(1)},\ldots,\widehat\bfe^{(K)}\}}\indic[o\in\Free(X,\bfe)]W(\bfe)$, we
have $q_o=(A_K+B_K)/(Z_K+\mathrm{tail}_K)$, $\hat q_o^{(K)}=A_K/Z_K$, so
$q_o-\hat q_o^{(K)}=[Z_KB_K-A_K\mathrm{tail}_K]/[Z_K(Z_K+\mathrm{tail}_K)]$; since
$0\le A_K\le Z_K$, $0\le B_K\le\mathrm{tail}_K$, the numerator is $\le Z_K\mathrm{tail}_K$
in absolute value, giving $|q_o-\hat q_o^{(K)}|\le\mathrm{tail}_K/(Z_K+\mathrm{tail}_K)$.
As $x\mapsto x/(Z_K+x)$ increases and $\mathrm{tail}_K\le\bar Z-Z_K$, this is
$\le(\bar Z-Z_K)/\bar Z$. (iii) $\bar Z$ is computed once in $O(n^2)$ worst-case time; after that, $\epsilon_K$ is updated in $O(1)$ time given $Z_K$ and is non-increasing in $K$.
\end{proof}

\begin{corollary}[Certificate for the direct-target marginal]\label{cor:qX-cert}
The same bound applies to $\widehat q_X^{(K)}$: $\;|q_X-\widehat q_X^{(K)}|\le\epsilon_K$.
\end{corollary}
\begin{proof}
The proof of Theorem~\ref{thm:cert} uses only that the marginalized quantity is an indicator in $[0,1]$; here the indicator is $\indic[\deg_X^+(\bfe)=0]$.
\end{proof}

\begin{corollary}[Operational action certificate]\label{cor:action-cert}
Let $a^\dagger := \arg\max_{a\in V_t}\widehat s_a^{(K)}$. If
\begin{equation}
\widehat s_{a^\dagger}^{(K)}- \max_{a\in V_t\setminus\{a^\dagger\}}\widehat s_a^{(K)} > 2\epsilon_K,
\end{equation}
then $a^\dagger = \arg\max_{a\in V_t} s_a$.
\end{corollary}
\begin{proof}
The certificate bounds every blocker marginal $\widehat q_o^{(K)}$ and, by Corollary~\ref{cor:qX-cert}, also $\widehat q_X^{(K)}$, each by the same $\epsilon_K$; the claim follows from the triangle inequality.
\end{proof}

\subsection{Operational certification flags and reporting}

For scene $s$, let
\[
\widehat s_{s,(1)}^{(K_s)}
\ge
\widehat s_{s,(2)}^{(K_s)}
\]
denote the largest and second-largest Top-$K_s$ action scores over
$a\in V_{t,s}$, including both the direct-target action and all
candidate blocker-removal actions. Define the approximate action margin
\begin{equation}
\widehat\Delta_{s}^{(K_s)}
=
\widehat s_{s,(1)}^{(K_s)}
-
\widehat s_{s,(2)}^{(K_s)}.
\label{eq:approx-action-margin}
\end{equation}

The argmax-certification flag is
\begin{equation}
\boxed{
c_{\mathrm{arg},s}
=
\mathbbm{1}
\left[
\widehat\Delta_s^{(K_s)}
>
2\epsilon_{K_s,s}
\right].
}
\label{eq:carg}
\end{equation}

When $c_{\mathrm{arg},s}=1$, the identity of the highest-scoring
physical action under the Top-$K$ approximation is identical to that
under the exact posterior of the proposed model.

On the complete Top-$K$-eligible evaluation set
$\mathcal S_{\mathrm{TopK}}$, the action certification rate is
\begin{equation}
\boxed{
\mathrm{CertRate}_{\mathrm{arg}}
=
\frac{1}{|\mathcal S_{\mathrm{TopK}}|}
\sum_{s\in\mathcal S_{\mathrm{TopK}}}
c_{\mathrm{arg},s}.
}
\label{eq:cert-rate}
\end{equation}

\noindent\textbf{Scope of the certificate.}
The certificate bounds Top-$K$ approximation error for every non-target marginal $q_o$ and, by Corollary~\ref{cor:qX-cert}, for the direct-target marginal $q_X$. It certifies approximation relative to the proposed posterior only; it does not certify perception correctness, physical grasp success, or operational safety.

\begin{remark}[Validity across all cycle lengths]
\label{rmk:cycleorder}
Proposition~\ref{prop:zbar} is \emph{not} a claim that only $2$-cycles are excluded. It uses only the implication ``acyclic $\Rightarrow$ no $2$-cycle'' to obtain an \emph{upper} bound. Cycles of length $\ge3$ remove \emph{additional} configurations from $\cD$, shrinking it further; hence $Z\le\bar Z$ remains valid (merely looser). The bound never enumerates cycle types. A tighter bound excluding $3$-cycles is possible, but $3$-cycle events share edges and are not independent, forfeiting the product form; we retain the two-factor bound as the simplest valid certificate. Consequently, when $\bar Z>Z$, $\epsilon_K$ has the positive floor $(\bar Z-Z)/\bar Z$, and the certificate may leave a stable argmax or free-set membership uncertified even after all practically available modes have been enumerated. This conservativeness affects stopping and certification status; it does not by itself trigger defer and is not a physical-safety guarantee. Action selection remains governed by the decision rule~\eqref{eq:policy}.
\end{remark}

\begin{remark}[Order-robustness and edge cases]
\label{rmk:cert}
Clip candidate-edge probabilities $p_{ij}\in[\zeta,1-\zeta]$ (e.g., $\zeta=10^{-4}$) before computation. Because the empty graph belongs to $\cD$, the feasible set is nonempty. Moreover, the empty graph has strictly positive weight after clipping, so $Z>0$. The certificate needs only $K$ distinct feasible configurations, not the $K$
largest, so it is immune to solver-ordering errors --- ordering affects only how fast
$\epsilon_K$ shrinks. Geometric-decay heuristics, by contrast, estimate the tail from
a single ratio $r^{(k+1)}/r^{(k)}$; these are non-monotone in combinatorial spaces,
so a ratio at a plateau boundary underestimates the unexplored mass and gives no
valid guarantee. We therefore drop such heuristics.
\end{remark}

\paragraph{Stopping criterion (margin-relative).}
Stop at the first $K$ with $\epsilon_K \le \max(\epsilon_{\mathrm{target}},\ c_\delta\,\hat\delta_K)$,
or $K = K_{\max}$, or the ILP is infeasible, where $\hat\delta_K$ is the current decision margin
(top $\hat s$ minus runner-up) and $0 < c_\delta < 1/2$. The certificate does not promise a fixed $\epsilon$: it certifies the argmax
when $\epsilon_K < \hat\delta_K/2$, and certifies free-set membership of $o$ when
$|\hat q^{(K)}_o - \tau_{\mathrm{set}}| > \epsilon_K$.

% ============================================================
\section{Computing $\hat q_o^{(K)}$ from the Top-$K$}
\label{sec:compute-qo}
% ============================================================

\subsection{Free set computation}

For each $\hat\bfe^{(k)}$, compute $F^{(k)} = \Free(X, \hat\bfe^{(k)})$ in two steps:

\paragraph{Step A: Reachability.} BFS from $X$:
\begin{equation}
\Reach(X, \hat\bfe^{(k)}) = \{v \in V_t : \exists \text{ path } X \to v \text{ in } G(\hat\bfe^{(k)})\}
\end{equation}

\paragraph{Step B: Out-degree filter.}
\begin{equation}
F^{(k)} = \{v \in \Reach(X, \hat\bfe^{(k)}) \setminus \{X\} : \deg^+_v(\hat\bfe^{(k)}) = 0\}
\end{equation}

Complexity: $O(n^2)$ per graph.

\subsection{Log-space numerical stability}

To avoid underflow of $W^{(k)} = \exp(\ell^{(k)})$, work with relative weights:
\begin{equation}
\ell_{\max} := \max_{k} \ell^{(k)}, \quad r^{(k)} := \exp(\ell^{(k)} - \ell_{\max})
\label{eq:relative}
\end{equation}
Properties: $r^{(1)} = 1$ (if $\ell^{(1)} = \ell_{\max}$); $r^{(k)} \in (0, 1]$.

\subsection{Final formula for $\hat q_o^{(K)}$}

Divide the numerator/denominator of \eqref{eq:qo-topk} by $\exp(\ell_{\max})$:

\begin{tcolorbox}[colback=green!5, colframe=green!50!black]
\begin{equation}
\hat q_o^{(K)} = \frac{\sum_{k=1}^K \indic[o \in F^{(k)}] \cdot r^{(k)}}{\sum_{k=1}^K r^{(k)}}
\label{eq:qhat-final}
\end{equation}
\end{tcolorbox}

% ============================================================
\subsection{MAP plug-in is an approximation of the Bayes-optimal decision}
\label{sec:map-vs-marg}
The Bayes-optimal decision \eqref{eq:ostar-true} \emph{defines} the correct action as
$\arg\max_a s_a$ over the \emph{entire} posterior. The MAP plug-in --- picking the most likely
structure $G^*=\arg\max_\bfe \PD(\bfe\mid\Omega)$ and reading off $\Free(G^*)$ --- is only an
\emph{approximation} using a single mode. When the posterior is diffuse (common in clutter),
this approximation can choose wrongly. We state clearly the two ways MAP fails; the reason for preferring
the marginal is \emph{Bayes optimality}, not any notion of ``stability'' or ``reliability''
(which would require stronger assumptions that the framework does not carry).

\begin{proposition}[Marginal integration uses the full posterior]
Consider the posterior distribution over the space of DAG structures:
$$P_{\cD}(G\mid\Omega).$$
For each object $  o  $, MAP plug-in returns the \emph{set} $\mathrm{Free}(G^*)$ with no ranking; any selection rule within it is arbitrary. Marginal integration computes:
$$q_o = \sum_G \mathbf{1}[o \in \mathrm{Free}(G)] \, P_{\cD}(G\mid\Omega).$$
\textbf{(a)} When $  G^* $ contains several free objects, MAP only reports the set of graspable objects without ranking information among them.
\textbf{(b)} Even when $  G^* $ contains a single clearly free object, MAP can still choose wrongly relative to marginalization. This happens when another object (e.g. $  o_2  $) is supported by many structures whose total posterior mass exceeds $  G^* $, even though no single structure supporting $o_2$ beats $G^*$. Then:
$$q_{o_2} > q_{o_1} \quad \text{although} \quad \mathbf{1}[o_1 \in \mathrm{Free}(G^*)] = 1, \quad \mathbf{1}[o_2 \in \mathrm{Free}(G^*)] = 0.$$
\textbf{(c)} In general, the marginal difference
$$q_{o_1} - q_{o_2} = \Pr(o_1 \in \mathrm{Free}, o_2 \notin \mathrm{Free} \mid \Omega) - \Pr(o_2 \in \mathrm{Free}, o_1 \notin \mathrm{Free} \mid \Omega)$$
is exactly the posterior mass on structures where the two objects have different free status. Hence marginalization exploits the whole posterior rather than a single local mode as MAP does.
\end{proposition}
\textit{Purpose.}
This proposition makes clear that MAP inference, although it picks the highest-probability structure, still exploits information from a single mode and can lead to biased decisions when the posterior has several significant modes --- a situation common in clutter. Marginalization, by taking an expectation over the entire posterior, yields an object-level ranking better aligned with the Bayes decision principle \eqref{eq:ostar-true}.

\section{Related work: M-best MAP}
% #####################################################################
Our inference stage instantiates classical M-best MAP machinery: sequential
exclusion via constraints (Nilsson, 1998), LP/ILP formulations (Fromer \& Globerson, 2009),
diversity-augmented variants (Batra et al., 2012); approximating marginals from enumerated
modes follows M-best marginals (Yanover \& Weiss, 2004; Flerova et al., 2016). We claim no
novelty in this machinery. The method-specific additions are (i) a tail-model-free,
per-scene computable stopping certificate (Thm~\ref{thm:cert}) from the feasible-set
structure, and (ii) a decision layer consuming Top-$K$ marginal estimates through cost-derived free-set thresholding and defer, together with optional margin-based certification of the resulting decision.

\section{Theory-to-evaluation interface}
\label{sec:protocol}
The numerical evaluation is specified in the separate locked evaluation protocol.  The
mathematical implementation must expose the following scene-indexed quantities without
test-time tuning:
\begin{enumerate}[leftmargin=1.6em,itemsep=2pt]
\item the fixed candidate-pair IDs and ground-truth labels;
\item raw and calibrated VLM scores on the same VLM-supported edge population;
\item geometric scores, signed depth evidence, and source-availability flags;
\item fused probabilities $p_{ij}$ used unchanged by every structural ablation;
\item product-model mass outside DAG support, exact marginals on the predefined exact
subset, Top-$K$ marginals, $\epsilon_K$, stopping $K$, and runtime;
\item action scores, the separately frozen thresholds $\tau_{\mathrm{act}}$ and
$\tau_{\mathrm{set}}$, the selected action, and the predicted free set.
\end{enumerate}
All data splits are by scene.  UOAIS-FT weights, geometric parameters, VLM calibration,
fusion weights, candidate rules, Top-$K$ settings, and decision thresholds are frozen
before test evaluation.  Structural comparisons receive identical fused probabilities;
only the inference rule changes.

\subsection{Appendix-only graph diagnostics}

These diagnostics localize failure mechanisms but do not replace Free F1 or RSR.  Let
$\widehat\bfe_s$ be a graph decoded with a validation-frozen rule and let
$\bfe_s^*$ be the ground-truth obstruction graph.

\paragraph{Reachability diagnostic.}
Define
\begin{equation}
\widehat{\mathcal R}_s=\Reach(X_s,\widehat\bfe_s)\setminus\{X_s\},
\qquad
\mathcal R_s^*=\Reach(X_s,\bfe_s^*)\setminus\{X_s\}.
\end{equation}
Reachability precision, recall, and F1 are ordinary set metrics between
$\widehat{\mathcal R}_s$ and $\mathcal R_s^*$.  A false local edge can enlarge
$\widehat{\mathcal R}_s$ even when the decoded graph remains acyclic.

\paragraph{Depth-inconsistency diagnostic.}
For frozen tolerances $\tau_r$ and $\tau_z$, define
\begin{equation}
\mathrm{DepthInc}_s
=
\frac{
\sum_{(i,j)}\widehat e_{s,ij}
\indic[r_{s,ij}\ge\tau_r]
\indic[\Delta z_{s,ij}< -\tau_z]
}{
\max\!\left(
\sum_{(i,j)}\widehat e_{s,ij}\indic[r_{s,ij}\ge\tau_r],1
\right)
}.
\label{eq:depth-inc}
\end{equation}
This measures decoded edges that contradict reliable depth ordering; it is not a
substitute for edge F1 because a depth-consistent edge can still be false.

\paragraph{Cycle diagnostics.}
The decoded any-cycle indicator is
\begin{equation}
\mathrm{AnyCycle}_s=\indic[\widehat\bfe_s\notin\cD],
\end{equation}
and a decoded two-cycle count is
\begin{equation}
N_{2\mathrm{cyc},s}
=
\sum_{i<j}\widehat e_{s,ij}\widehat e_{s,ji}.
\end{equation}
These inspect one decoded graph.  By contrast, $\mu_s$ measures probability mass outside
DAG support under the entire product model, so the two quantities must not be conflated.

% ============================================================
\section{Overall algorithm and checklist}
% ============================================================

\subsection{Algorithm}

\subsubsection*{Algorithm 1: Reliability-aware Top-$K$ obstruction reasoning}
The frozen offline parameters are $\theta_{\mathrm{FT}}$,
$(b^{cv},\kappa_o,\kappa_c,\kappa_z,o_0,c_0,\rho,\sigma_z)$,
$\Theta_{\mathrm{cal}}$, $(\gamma,\beta_{vlm},\beta_{cv})$, $\zeta$,
$K_{\max}$, $\epsilon_{\mathrm{target}}$, $c_\delta$,
$\tau_{\mathrm{act}}$, and $\tau_{\mathrm{set}}$.
\begin{algorithmic}[1]
\Require RGB-D scene $I=(I_{rgb},I_d)$, instruction $p$, target $X$, object set $V_t$,
and fixed pair universe $\cP_{\mathrm{cand}}$.
\If{$V_t=\{X\}$} \Return \textsf{grasp $X$ directly} \EndIf

\State \textbf{Stage 0 --- UOAIS-FT geometric perception.}
\State $\{V_i,A_i,O_i\}_{i\in V_t}\gets f_{\theta_{\mathrm{FT}}}(I_{rgb},I_d)$

\State \textbf{Stage 1 --- Independent per-edge sources and fusion.}
\For{each $(i,j)\in\cP_{\mathrm{cand}}$}
    \State $IV_i\gets A_i\setminus V_i$
    \State $C^{hid}_{ij}\gets IV_i\cap V_j$;
           $R_i^{clr}\gets\Dilate(A_i,\rho)\setminus A_i$;
           $C^{clr}_{ij}\gets R_i^{clr}\cap V_j$
    \State $o_{ij}\gets |C^{hid}_{ij}|/\max(|A_i|,1)$;
           $c_{ij}\gets |C^{clr}_{ij}|/\max(|R_i^{clr}|,1)$
    \State compute $z_i,z_j,r_{ij}$ by~\eqref{eq:median-depth}--\eqref{eq:depth-reliability};
           $d_{ij}\gets r_{ij}\tanh(\Delta z_{ij}/\max(\sigma_z,\varepsilon_z))$
    \State $u^{cv}_{ij}\gets b^{cv}+\kappa_o(o_{ij}-o_0)+\kappa_c(c_{ij}-c_0)+\kappa_zd_{ij}$;
           $p^{cv}_{ij}\gets\sigma(u^{cv}_{ij})$
    \State $\delta^{cv}_{ij}\gets\indic[\phi^{cv}_{ij}\text{ computable}]$
    \State independently query $R^{vlm}_{ij}\gets f_{\mathrm{VLM}}(I_{rgb},p,i,j)$
           \Comment{no geometric hint}
    \State compute raw $p^{vlm}_{ij}$ and calibrated $p^{vlm,cal}_{ij}$ by~\eqref{eq:vlm-cal}
    \State $\delta^{vlm}_{ij}\gets\indic[p^{vlm}_{ij}\text{ valid}]$
    \State $\lambda^{cv}_{ij}\gets u^{cv}_{ij}-\logit\pi_0$;
           $\lambda^{vlm}_{ij}\gets\logit(p^{vlm,cal}_{ij})-\logit\pi_0$
    \State $g_{ij}\gets\gamma+\delta^{vlm}_{ij}\beta_{vlm}\lambda^{vlm}_{ij}
           +\delta^{cv}_{ij}\beta_{cv}\lambda^{cv}_{ij}$
    \State $p_{ij}\gets\operatorname{clip}(\sigma(g_{ij}),\zeta,1-\zeta)$;
           $w_{ij}\gets\logit(p_{ij})$
\EndFor

\State \textbf{Stage 2 --- Top-$K$ feasible DAG enumeration.}
\State $\log\bar Z\gets\sum_{i<j}\log(1-p_{ij}p_{ji})$, taking absent directed pairs as probability $0$
\State $\mathrm{solutions}\gets\varnothing$, $k\gets0$
\Repeat
    \State solve the ILP~\eqref{eq:ilp-obj} with lazy cycle cuts and all previous no-good cuts
    \If{infeasible} \textbf{break} \EndIf
    \State $k\gets k+1$; extract $\hat\bfe^{(k)}$ and $\ell^{(k)}$
    \State $F^{(k)}\gets\Free(X,\hat\bfe^{(k)})$; $Y_X^{(k)}\gets\indic[\deg_X^+(\hat\bfe^{(k)})=0]$
    \State append $(\hat\bfe^{(k)},\ell^{(k)},F^{(k)},Y_X^{(k)})$ to $\mathrm{solutions}$
    \State compute relative weights $r^{(h)}=\exp(\ell^{(h)}-\ell^{(1)})$, $h\le k$
    \State update $\widehat q_o^{(k)}$ by~\eqref{eq:qhat-final} and $\widehat q_X^{(k)}$ by~\eqref{eq:qX-topk}
    \State $\widehat s_X^{(k)}\gets\widehat q_X^{(k)}$ and $\widehat s_o^{(k)}\gets\widehat q_o^{(k)}$
    \State $\widehat\delta_k\gets\widehat s_{(1)}^{(k)}-\widehat s_{(2)}^{(k)}$
    \State $\log Z_k\gets\LSE(\ell^{(1)},\ldots,\ell^{(k)})$;
           $\epsilon_k\gets\max\{0,1-\exp(\log Z_k-\log\bar Z)\}$
\Until{$\epsilon_k\le\max(\epsilon_{\mathrm{target}},c_\delta\widehat\delta_k)$
       \textbf{ or } $k=K_{\max}$}
\State \textbf{assert}$(\mathrm{solutions}\neq\varnothing)$

\State \textbf{Stage 3 --- Decisions.}
\State $\widehat F^{(K)}\gets\{o:\widehat q_o^{(k)}>\tau_{\mathrm{set}}\}$
\State $\widehat a_{raw}\gets\arg\max_{a\in V_t}\widehat s_a^{(k)}$
\If{$\widehat s_{\widehat a_{raw}}^{(k)}>\tau_{\mathrm{act}}$}
    \State $\mathrm{action}\gets\widehat a_{raw}$
\Else
    \State $\mathrm{action}\gets\textsf{defer}$
\EndIf
\State $c_{arg}\gets\indic[\widehat\delta_k>2\epsilon_k]$
\State $c_{mem}^{(o)}\gets\indic[|\widehat q_o^{(k)}-\tau_{\mathrm{set}}|>\epsilon_k]$ for every $o$
\State \Return $(\mathrm{action},\widehat F^{(K)},k,\epsilon_k,c_{arg},\{c_{mem}^{(o)}\})$
\end{algorithmic}
