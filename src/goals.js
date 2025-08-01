/**
 * WidowX AI (wxai_base.urdf) demo             – closed-chain-ik + Three.js
 * Drag the blue gizmo to move the end-effector (ee_gripper_link).
 * If you host the URDF / meshes locally, just update the two constants
 *   WXAI_URDF_PATH  and  PACKAGE_ROOT  below.
 */

/* ─────────────────────────────────  Imports  ────────────────────────────── */
import {
  WebGLRenderer,
  PerspectiveCamera,
  Color,
  Scene,
  DirectionalLight,
  AmbientLight,
  Group,
  Vector3,
  Mesh,
  PCFSoftShadowMap,
} from 'three';
import { OrbitControls }    from 'three/examples/jsm/controls/OrbitControls.js';
import { TransformControls }from 'three/examples/jsm/controls/TransformControls.js';
import { GUI }              from 'three/examples/jsm/libs/lil-gui.module.min.js';
import Stats                from 'three/examples/jsm/libs/stats.module.js';
import {
  Solver,
  Link,
  Joint,
  SOLVE_STATUS_NAMES,
  IKRootsHelper,
  setUrdfFromIK,
  urdfRobotToIKRoot,
  setIKFromUrdf,
  Goal,
  DOF,
  SOLVE_STATUS,
} from 'closed-chain-ik';
import URDFLoader           from 'urdf-loader';
import { STLLoader }        from 'three/examples/jsm/loaders/STLLoader.js';
import { ColladaLoader }    from 'three/examples/jsm/loaders/ColladaLoader.js';

/* ───────────────────────────  WidowX-specific paths  ────────────────────── */
const WXAI_URDF_PATH =                                           // CHANGED
  '../public/assets/' +         // CHANGED
  'trossen_arm_description/main/urdf/generated/wxai/wxai_base.urdf'; // CHANGED
const PACKAGE_ROOT   =                                           // CHANGED
  '../public/assets/trossen_arm_description'; // CHANGED

/* ─────────────────────────────  UI / solver params  ─────────────────────── */
const params = {
  animate: 'none',                                               // CHANGED
  baseTilt: 0,
  solve: true,
  displayMesh: true,
  displayIk: false,
  enableControls: true,
  settleIterations: 6,
  displayConvergedOnly: true,
};

const solverOptions = {
  maxIterations: 10,
  divergeThreshold: 0.05,
  stallThreshold: 1e-5,
  translationErrorClamp: 0.01,
  rotationErrorClamp: 0.01,
  translationConvergeThreshold: 1e-5,
  rotationConvergeThreshold: 1e-5,
};

/* ──────────────────────  Globals (platform* → ee*)  ─────────────────────── */
let gui, stats;
let outputContainer, renderer, scene, camera;
let controls, transformControls, targetObject, directionalLight;
let ikNeedsUpdate = true;
const tempVec = new Vector3();

let urdfRoot, ikRoot, ikHelper, drawThroughIkHelper, solver;
let eeLink, eeGoal;                                              // CHANGED

/* ───────────────────────────────  Bootstrap  ────────────────────────────── */
init();
render();

/* ────────────────────────────────  init()  ──────────────────────────────── */
function init() {

  stats = new Stats();
  document.body.appendChild( stats.dom );
  outputContainer = document.getElementById( 'output' );

  /* --- Renderer --- */
  renderer = new WebGLRenderer( { antialias: true } );
  renderer.setPixelRatio( window.devicePixelRatio );
  renderer.setSize( window.innerWidth, window.innerHeight );
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = PCFSoftShadowMap;
  document.body.appendChild( renderer.domElement );

  /* --- Camera & scene --- */
  camera = new PerspectiveCamera( 50, window.innerWidth / window.innerHeight, 0.01, 100 );
  camera.position.set( 0.3, 0.3, 0.3 );

  scene = new Scene();
  scene.background = new Color( 0x1e161d );

  /* --- Lights --- */
  directionalLight = new DirectionalLight();
  directionalLight.position.set( 1, 3, 2 );
  directionalLight.intensity = 3 * 0.75;
  directionalLight.castShadow = true;
  directionalLight.shadow.mapSize.setScalar( 2048 );

  const sCam = directionalLight.shadow.camera;
  sCam.top = sCam.right = 0.25; sCam.left = sCam.bottom = -0.25;
  sCam.near = 0; sCam.far = 10; sCam.updateProjectionMatrix();

  const backLight = new DirectionalLight();
  backLight.intensity = 3 * 0.25;
  backLight.position.set( -1, -3, -2 );

  const ambientLight = new AmbientLight( 0x1f1a1e, 3 );
  scene.add( directionalLight, directionalLight.target, backLight, ambientLight );

  /* --- Orbit & gizmo controls --- */
  controls = new OrbitControls( camera, renderer.domElement );
  controls.target.y = 0.05;
  controls.update();

  transformControls = new TransformControls( camera, renderer.domElement );
  transformControls.setSpace( 'local' );
  scene.add( transformControls.getHelper() );

  transformControls.addEventListener( 'mouseDown', () => {
    controls.enabled = false;
    params.animate = 'none';                                     // CHANGED
  } );

  transformControls.addEventListener( 'mouseUp', () => {
    controls.enabled = true;
    ikNeedsUpdate = true;
    setIKFromUrdf( ikRoot, urdfRoot );
    ikRoot.updateMatrixWorld( true );
  } );

  /* --- Target object (drives ee_goal) --- */
  targetObject = new Group();
  scene.add( targetObject );
  transformControls.attach( targetObject );
  transformControls.addEventListener( 'objectChange', () => {
    ikNeedsUpdate = true;
  } );

  /* --- GUI --- */
  gui = new GUI();
  gui.add( params, 'enableControls' );
  gui.add( params, 'animate', [ 'none', 'gyrate', 'rotate' ] ).listen();
  gui.add( params, 'baseTilt', -0.3, 0.3, 1e-4 ).onChange( () => ikNeedsUpdate = true );
  gui.add( params, 'solve' );
  gui.add( params, 'displayMesh' );
  gui.add( params, 'displayIk' );
  gui.add( params, 'displayConvergedOnly' );
  gui.add( params, 'settleIterations', 1, 20, 1 ).onChange( () => ikNeedsUpdate = true );

  /* --- URDF loader  --- */
  const loader = new URDFLoader();
  loader.fetchOptions = { mode: 'cors' };
  loader.packages = {                                             // CHANGED
    'trossen_arm_description': PACKAGE_ROOT                       // CHANGED
  };

  loader.loadMeshCb = ( path, manager, done ) => {
    if ( /\.stl$/.test( path ) ) {
      new STLLoader( manager ).load( path, res => {
        const m = new Mesh( res );
        m.castShadow = m.receiveShadow = true; done( m );
      } );
    } else if ( /\.dae$/.test( path ) ) {
      new ColladaLoader( manager ).load( path, res => {
        const model = res.scene;
        const lights = [];
        model.traverse( c => { c.castShadow = c.receiveShadow = true; if ( c.isLight ) lights.push( c ); } );
        lights.forEach( l => l.parent.remove( l ) ); done( model );
      } );
    }
  };

  /* --- Load WidowX URDF  --- */
  loader.loadAsync( WXAI_URDF_PATH )                              // CHANGED
    .then( result => {

      urdfRoot = result;

      /* make floating joints fixed; make revolutes continuous for demo */
      urdfRoot.traverse( c => { if ( c.jointType === 'floating' ) c.jointType = 'fixed'; } );
      for ( const j of Object.values( urdfRoot.joints ) ) {
        if ( j.jointType === 'revolute' ) j.jointType = 'continuous'; // CHANGED
      }

      /* IK root */
      ikRoot = urdfRobotToIKRoot( urdfRoot );
      ikRoot.setDoF();

      /* ROS (Z-up) → Three.js (Y-up) */
      urdfRoot.rotation.set( -Math.PI / 2, 0, 0 );
      ikRoot.setEuler( -Math.PI / 2, 0, 0 );

      /* --- End-effector goal (ee_gripper_link) --- */
      eeLink = ikRoot.find( n => n.name === 'ee_gripper_link' );  // CHANGED
      eeGoal = new Goal();                                        // CHANGED
      eeGoal.setEuler( -Math.PI / 2, 0, 0 );                      // CHANGED
      eeGoal.setPosition( 0.35, 0.15, 0.0 );                      // CHANGED
      eeGoal.makeClosure( eeLink );                               // CHANGED

      targetObject.quaternion.set( ...eeGoal.quaternion );        // CHANGED
      targetObject.position.set( ...eeGoal.position );            // CHANGED

      /* helpers */
      ikHelper          = new IKRootsHelper( [ ikRoot, eeGoal ] );// CHANGED
      drawThroughIkHelper = new IKRootsHelper( [ ikRoot, eeGoal ] );// CHANGED
      ikHelper.setColor( 0x2196f3 );                              // CHANGED
      drawThroughIkHelper.setColor( 0x2196f3 );                   // CHANGED
      ikHelper.setJointScale( 0.03 ); drawThroughIkHelper.setJointScale( 0.03 );
      drawThroughIkHelper.setDrawThrough( true );

      solver = new Solver( [ ikRoot, eeGoal ] );                  // CHANGED
      scene.add( urdfRoot, ikHelper, drawThroughIkHelper );
    } );

  /* --- Resize & keybindings --- */
  window.addEventListener( 'resize', () => {
    renderer.setSize( window.innerWidth, window.innerHeight );
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
  } );

  window.addEventListener( 'keydown', e => {
    switch ( e.key ) {
      case 'w': transformControls.setMode( 'translate' ); break;
      case 'e': transformControls.setMode( 'rotate'    ); break;
      case 'q': transformControls.setSpace( transformControls.space === 'local' ? 'world' : 'local' ); break;
      case 'f': controls.target.set( 0, 0.05, 0 ); controls.update(); break;
    }
  } );

}

/* ─────────────────────────────  IK solver  ─────────────────────────────── */
function updateIk() {

  eeGoal.setPosition(                                            // CHANGED
    targetObject.position.x, targetObject.position.y, targetObject.position.z );
  eeGoal.setQuaternion(                                          // CHANGED
    targetObject.quaternion.x, targetObject.quaternion.y,
    targetObject.quaternion.z, targetObject.quaternion.w );

  let solveOutput = '', totalTime = 0;
  setIKFromUrdf( ikRoot, urdfRoot );

  let isConverged = false;
  for ( let i = 0; i < params.settleIterations; i ++ ) {

    ikRoot.updateMatrixWorld( true );
    Object.assign( solver, solverOptions );

    const t0 = performance.now();
    const res = solver.solve();
    const dt = performance.now() - t0;
    totalTime += dt;

    solveOutput += `${ dt.toFixed( 2 ) }ms ${ SOLVE_STATUS_NAMES[ res[ 0 ] ] }\n`;

    const allConv = res.every( r => r === SOLVE_STATUS.CONVERGED );
    const allDiv  = res.every( r => r === SOLVE_STATUS.DIVERGED  );
    const allStal = res.every( r => r === SOLVE_STATUS.STALLED   );
    if ( allConv || allDiv || allStal ) { isConverged = allConv; break; }

  }

  outputContainer.textContent = solveOutput + '\nTotal: ' + totalTime.toFixed( 2 ) + 'ms';
  if ( !params.displayConvergedOnly || isConverged ) setUrdfFromIK( urdfRoot, ikRoot );

}

/* ─────────────────────────────  Render loop  ───────────────────────────── */
function render() {

  requestAnimationFrame( render );

  if ( ikRoot && !transformControls.dragging ) {
    targetObject.matrix.set( ...eeLink.matrixWorld ).transpose(); // CHANGED
    targetObject.matrix.decompose( targetObject.position, targetObject.quaternion, targetObject.scale );

    const p = targetObject.position, q = targetObject.quaternion;
    eeGoal.setPosition( p.x, p.y, p.z );                         // CHANGED
    eeGoal.setQuaternion( q.x, q.y, q.z, q.w );                  // CHANGED
  }

  if ( urdfRoot ) {

    ikRoot.setEuler( -Math.PI / 2, 0, params.baseTilt );

    if ( params.animate === 'gyrate' ) {
      const t = performance.now() * 0.004;
      targetObject.position.set( Math.sin( t ) * 0.05, 0.15, Math.cos( t ) * 0.05 );
      targetObject.rotation.set( -Math.PI / 2, 0, 0 );
      ikNeedsUpdate = true;
    } else if ( params.animate === 'rotate' ) {
      const t = performance.now() * 0.004;
      targetObject.position.set( 0.35, 0.15 + Math.sin( t ) * 0.02, 0 );
      targetObject.rotation.set( -Math.PI / 2, 0, Math.cos( t ) * 0.75 );
      ikNeedsUpdate = true;
    }

    if ( ikNeedsUpdate && params.solve ) { updateIk(); ikNeedsUpdate = false; }

    urdfRoot.visible          = params.displayMesh;
    ikHelper.visible          = drawThroughIkHelper.visible = params.displayIk;

    tempVec.subVectors( directionalLight.position, directionalLight.target.position );
    directionalLight.target.position.copy( urdfRoot.position );
    directionalLight.position.copy( urdfRoot.position ).add( tempVec );
  }

  transformControls.enabled = transformControls.visible = params.enableControls;
  renderer.render( scene, camera );
  stats.update();
}
