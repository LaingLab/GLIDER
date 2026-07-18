document$.subscribe(() => {
  mermaid.initialize({ startOnLoad: false });

  const diagrams = document.querySelectorAll("article .mermaid");
  diagrams.forEach((diagram) => diagram.removeAttribute("data-processed"));
  mermaid.run({ nodes: diagrams });
});
