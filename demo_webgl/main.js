import * as THREE from 'three';

import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import Stats from 'three/examples/jsm/libs/stats.module'

const passthroughVS = /* glsl */`
varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`;

const MaskShader = {
    uniforms: {
        'tOriginal': { value: null },
        'threshold': { value: 0.5 },
    },
    vertexShader: passthroughVS,
    fragmentShader: /* glsl */`
		uniform sampler2D tOriginal;
        uniform float threshold;

		varying vec2 vUv;

		void main() {
            vec3 inpColor = texture2D( tOriginal, vUv ).xyz;
			gl_FragColor = vec4(vec3(clamp((inpColor.x - threshold)*140.0, 0.0, 1.0)), 1.0);
		}`
};

const CopyShader = {
    uniforms: {
        'tOriginal': { value: null },
    },
    vertexShader: passthroughVS,
    fragmentShader: /* glsl */`
		uniform sampler2D tOriginal;

		varying vec2 vUv;

		void main() {
            vec2 uv = vUv;
            vec3 inpColor = texture2D( tOriginal, uv ).xyz;
			gl_FragColor = vec4(inpColor, 1.0);
		}`
};



const FinalCombinePassShader = {
    uniforms: {
        't0': { value: null },
        't1': { value: null },

        'm0': { value: null },
        'm1': { value: null },
        'm2': { value: null },
        'm3': { value: null },
        'm4': { value: null },
        'm5': { value: null },
        'm6': { value: null },
        'm7': { value: null },

        'enable': { value: 0.5 },
        'zoom': { value: 2.0 },
        'mip': { value: 6.0 },
        'stride': { value: 1 },
        'aspectRatio' : { value: 1.0 },
    },
    vertexShader: passthroughVS,
    fragmentShader: /* glsl */`
        uniform sampler2D t0;
        uniform sampler2D t1;

        uniform sampler2D m0;
        uniform sampler2D m1;
        uniform sampler2D m2;
        uniform sampler2D m3;
        uniform sampler2D m4;
        uniform sampler2D m5;
        uniform sampler2D m6;
        uniform sampler2D m7;

        uniform float enable;
        uniform float zoom;
        uniform int mip;
        uniform int stride;

        uniform float aspectRatio;

        varying vec2 vUv;

        // The demo stores the mask pyramid in separate render targets.
        vec4 sampleMask(vec2 uv, int lod) {
            if (lod == 0) return texture2D(m1, uv);
            if (lod == 1) return texture2D(m2, uv);
            if (lod == 2) return texture2D(m3, uv);
            if (lod == 3) return texture2D(m4, uv);
            if (lod == 4) return texture2D(m5, uv);
            if (lod == 5) return texture2D(m6, uv);
            return texture2D(m7, uv);
        }

		void main() {
            vec2 uv = vUv / vec2(zoom);
            uv.x *= aspectRatio;

            vec4 fine0 = texture2D(t0, uv);
            vec4 fine1 = texture2D(t1, uv);
            if (enable < 1.0) {
                gl_FragColor = mix(fine1, fine0, sampleMask(uv, mip));
                return;
            }

            vec4 accum = vec4(0.0);
            int fine = 0;
            for (int band = 0; band < 6; band++) {
                if (fine >= mip) break;
                int coarse = fine + stride;
                if (coarse > mip) coarse = mip;
                vec4 coarse0 = texture2D(t0, uv, float(coarse));
                vec4 coarse1 = texture2D(t1, uv, float(coarse));
                accum += mix(fine1 - coarse1, fine0 - coarse0, sampleMask(uv, fine));
                fine = coarse;
                fine0 = coarse0;
                fine1 = coarse1;
            }
            gl_FragColor = accum + mix(fine1, fine0, sampleMask(uv, mip));
		}`
};


const params = {
    enable: true,
    threshold: 0.6,
    zoom: 2.5,
    mip: 5.0,
    stride: 1,
    tex0: 1,
    tex1: 3,
    save: saveAsImage,
};

let renderer, effectCopy, effectFinalCombine, effectMask, mips, textures, aspectRatio;

const stats = Stats()
document.body.appendChild(stats.dom)

init();

function init() {
    aspectRatio = window.innerWidth / window.innerHeight;
    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    renderer.toneMapping = THREE.LinearToneMapping;
    renderer.outputEncoding = THREE.LinearEncoding;

    let loaded = 0;
    mips = new Object();
    textures = new Object();
    let loadData = function (tex, name) {
        const material = new THREE.MeshBasicMaterial({ map: tex });
        if (name == "mask") {
            let w = tex.image.width;
            let h = tex.image.height;

            let sharedProps = { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, format: THREE.RGBAFormat, type: THREE.UnsignedByteType };

            let mips_l = []
            while (w > 1 && h > 1) {
                mips_l.push(new THREE.WebGLRenderTarget(w, h, sharedProps));
                w = w / 2;
                h = h / 2;
            }

            mips[name] = mips_l;
        }

        textures[name] = tex;

        loaded += 1;
        if (loaded == 7)
            render();
    }

    new THREE.TextureLoader().load("1.jpg", function ( texture ) {loadData(texture, "0")});
    new THREE.TextureLoader().load("2.jpg", function ( texture ) {loadData(texture, "1")});
    new THREE.TextureLoader().load("1rot.jpg", function ( texture ) {loadData(texture, "2")});
    new THREE.TextureLoader().load("2rot.jpg", function ( texture ) {loadData(texture, "3")});    
    new THREE.TextureLoader().load("normal0.jpg", function ( texture ) {loadData(texture, "4")});
    new THREE.TextureLoader().load("normal1.jpg", function ( texture ) {loadData(texture, "5")});
    new THREE.TextureLoader().load("perlin.jpg", function ( texture ) {loadData(texture, "mask")});

    effectCopy = new ShaderPass(CopyShader);
    effectFinalCombine = new ShaderPass(FinalCombinePassShader);
    effectMask = new ShaderPass(MaskShader, null);

    const gui = new GUI();

    gui.add(params, 'enable');
    gui.add(params, 'threshold', 0.0, 1.0);
    gui.add(params, 'zoom', 1.0, 4.0);
    gui.add(params, 'mip', 0, 6, 1.0);
    gui.add(params, 'stride', 1, 2, 1).name('Passo dos níveis');
    gui.add(params, 'tex0', { Texture0: 0, Texture1: 1, Texture0Rotated: 2, Texture1Rotated: 3, Normal0: 4, Normal1: 5});
    gui.add(params, 'tex1', { Texture0: 0, Texture1: 1, Texture0Rotated: 2, Texture1Rotated: 3, Normal0: 4, Normal1: 5});
    gui.add(params, 'save');

    gui.open();
}

window.addEventListener('resize', onWindowResize, false);

function saveAsImage() {
    render();
    var imgData;
    var strDownloadMime = "image/octet-stream";
    var strMime = "image/jpeg";
    imgData = renderer.domElement.toDataURL(strMime);
    imgData = imgData.replace(strMime, strDownloadMime);
    var link = document.createElement('a');
    if (typeof link.download === 'string') {
        document.body.appendChild(link);
        link.download = 'snapshot.jpg';
        link.href = imgData;
        link.click();
        document.body.removeChild(link); //remove the link when done
    } else {
        location.replace(uri);
    }
}

function onWindowResize() {
    aspectRatio = window.innerWidth / window.innerHeight;
    renderer.setSize(window.innerWidth, window.innerHeight);
    render();
}

function render() {
    effectMask.uniforms['threshold'].value = params.threshold;
    effectMask.uniforms['tOriginal'].value = textures["mask"];
    effectMask.render(renderer, mips["mask"][0]);

    for (let i = 0; i < 7; i++) {
        effectCopy.uniforms['tOriginal'].value = mips["mask"][i].texture;
        effectCopy.render(renderer, mips["mask"][i + 1]);
    }

    effectFinalCombine.renderToScreen = true;
    for (let i = 0; i < 8; i++) { 
        effectFinalCombine.uniforms['m'+i].value = mips["mask"][i].texture;
    }
    effectFinalCombine.uniforms['mip'].value = params.mip;
    effectFinalCombine.uniforms['stride'].value = params.stride;
    effectFinalCombine.uniforms['t0'].value = textures[params.tex0];
    effectFinalCombine.uniforms['t1'].value = textures[params.tex1];
    effectFinalCombine.uniforms['enable'].value = params.enable;
    effectFinalCombine.uniforms['zoom'].value = params.zoom;
    effectFinalCombine.uniforms['aspectRatio'].value = aspectRatio;
    effectFinalCombine.render(renderer, null);

    requestAnimationFrame(render);
    stats.update();
}
