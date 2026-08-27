/**
 * A quiet, living map of the knowledge base. The visual language is deliberately
 * cartographic rather than sci-fi: contours are documents, routes are retrieval
 * paths, and the pulsing points are citations coming into focus.
 */
export function AmbientKnowledgeField() {
  return (
    <div className="knowledge-field" aria-hidden="true">
      <div className="knowledge-aurora knowledge-aurora--clay" />
      <div className="knowledge-aurora knowledge-aurora--sage" />
      <div className="knowledge-aurora knowledge-aurora--gold" />

      <svg
        className="knowledge-atlas"
        viewBox="0 0 1200 760"
        preserveAspectRatio="xMidYMid slice"
        focusable="false"
      >
        <g className="knowledge-contours knowledge-contours--far">
          <path d="M-120 164C38 36 215 31 338 124c104 79 72 207 189 268 142 74 260-76 405-25 139 49 172 174 388 111" />
          <path d="M-105 205C45 92 201 83 309 156c96 66 73 179 177 241 136 81 259-46 401-6 151 43 203 154 428 102" />
          <path d="M-86 246C53 150 190 137 282 190c88 52 80 151 169 211 127 86 254-16 393 12 164 34 236 130 470 88" />
          <path d="M-63 287C61 209 180 190 258 225c79 35 86 122 148 179 111 101 246 14 382 31 179 22 270 102 512 75" />
        </g>

        <g className="knowledge-contours knowledge-contours--near">
          <path d="M716-90c-93 86-131 177-73 259 66 92 205 50 274 140 63 83 8 194 83 268 70 69 181 29 267 104" />
          <path d="M764-82c-78 78-108 155-57 225 62 84 189 50 254 129 62 75 18 174 85 242 62 62 157 40 244 107" />
          <path d="M814-74c-65 69-87 132-44 191 58 79 172 51 235 116 61 63 28 155 88 216 54 56 134 50 219 109" />
          <path d="M865-65c-51 59-66 109-31 158 52 72 155 51 215 103 60 52 37 136 90 191 47 49 111 59 195 109" />
        </g>

        <g className="knowledge-route">
          <path d="M91 582C249 487 332 555 462 470s247-159 392-96 177-57 275-142" />
          <path d="M163 116c126 42 156 133 275 139 127 6 176-82 300-42 106 35 169 141 314 119" />
        </g>

        <g className="knowledge-nodes">
          <g className="knowledge-node knowledge-node--1" transform="translate(161 541)">
            <circle className="knowledge-node__halo" r="18" />
            <circle className="knowledge-node__core" r="3.5" />
          </g>
          <g className="knowledge-node knowledge-node--2" transform="translate(420 495)">
            <circle className="knowledge-node__halo" r="13" />
            <circle className="knowledge-node__core" r="3" />
          </g>
          <g className="knowledge-node knowledge-node--3" transform="translate(684 392)">
            <circle className="knowledge-node__halo" r="20" />
            <circle className="knowledge-node__core" r="4" />
          </g>
          <g className="knowledge-node knowledge-node--4" transform="translate(1005 319)">
            <circle className="knowledge-node__halo" r="15" />
            <circle className="knowledge-node__core" r="3" />
          </g>
          <g className="knowledge-node knowledge-node--5" transform="translate(443 255)">
            <circle className="knowledge-node__halo" r="12" />
            <circle className="knowledge-node__core" r="2.8" />
          </g>
          <g className="knowledge-node knowledge-node--6" transform="translate(832 244)">
            <circle className="knowledge-node__halo" r="17" />
            <circle className="knowledge-node__core" r="3.5" />
          </g>
        </g>
      </svg>

      <div className="knowledge-scan" />
      <div className="knowledge-grain" />
      <div className="knowledge-vignette" />
    </div>
  );
}
