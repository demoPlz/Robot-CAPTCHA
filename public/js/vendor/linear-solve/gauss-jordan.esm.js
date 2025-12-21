/**
 * ES Module wrapper for linear-solve (CommonJS)
 */

// Import the CommonJS module
const linearSolveModule = (function(){
/**
 * Used internally to solve systems
 * If you want to solve A.x = B,
 * choose data=A and mirror=B.
 * mirror can be either an array representing a vector
 * or an array of arrays representing a matrix.
 */
function Mat(data, mirror) {
  // Clone the original matrix
  this.data = new Array(data.length);
  for (var i=0, cols=data[0].length; i<data.length; i++) {
    this.data[i] = new Array(cols);
    for(var j=0; j<cols; j++) {
      this.data[i][j] = data[i][j];
    }
  }
  // Clone the mirror vector/matrix
  this.mirror = new Array(mirror.length);
  if (mirror.length && mirror[0].length) {
    for (var i=0; i<mirror.length; i++) {
      this.mirror[i] = new Array(mirror[0].length);
      for (var j=0; j<mirror[0].length; j++) {
        this.mirror[i][j] = mirror[i][j];
      }
    }
  }
  else {
    for (var i=0; i<mirror.length; i++) {
      this.mirror[i] = mirror[i];
    }
  }
}

Mat.prototype.gauss = function gauss() {
  var d = this.data;
  var n = this.data.length;
  var m = this.data[0].length;

  var pivots = new Array(n);
  var exchanges = new Array(n);
  for (var i=0; i<n; i++) {
    pivots[i] = i;
    exchanges[i] = true;
  }
  
  for (var i=0; i<n; i++) {
    var row = i;
    // Find pivot
    for (var j=i; j<n; j++) {
      if (!exchanges[j]) continue;
      if (Math.abs(d[pivots[j]][i]) > Math.abs(d[pivots[row]][i])) {
        row = j;
      }
    }
    if (Math.abs(d[pivots[row]][i]) < 1e-9) {
      throw new Error("The system doesn't have a unique solution");
    }
    // Swap pivot rows
    var tmp = pivots[i];
    pivots[i] = pivots[row];
    pivots[row] = tmp;

    var pivot = d[pivots[i]][i];
    for (var j=i+1; j<m; j++) {
      d[pivots[i]][j] /= pivot;
    }
    if (this.mirror.length && this.mirror[0].length) {
      for (var j=0; j<this.mirror[0].length; j++) {
        this.mirror[pivots[i]][j] /= pivot;
      }
    }
    else {
      this.mirror[pivots[i]] /= pivot;
    }
    d[pivots[i]][i] = 1;
    
    // Eliminate column
    for (var j=0; j<n; j++) {
      if (j === i) continue;
      var coeff = d[pivots[j]][i];
      for (var k=i+1; k<m; k++) {
        d[pivots[j]][k] -= d[pivots[i]][k] * coeff;
      }
      if (this.mirror.length && this.mirror[0].length) {
        for (var k=0; k<this.mirror[0].length; k++) {
          this.mirror[pivots[j]][k] -= this.mirror[pivots[i]][k] * coeff;
        }
      }
      else {
        this.mirror[pivots[j]] -= this.mirror[pivots[i]] * coeff;
      }
      d[pivots[j]][i] = 0;
    }
  }
  
  // Unscramble the rows
  var unscrambled;
  if (this.mirror.length && this.mirror[0].length) {
    unscrambled = new Array(this.mirror.length);
    for (var i=0; i<this.mirror.length; i++) {
      unscrambled[i] = new Array(this.mirror[0].length);
      for (var j=0; j<this.mirror[0].length; j++) {
        unscrambled[i][j] = this.mirror[pivots[i]][j];
      }
    }
  }
  else {
    unscrambled = new Array(this.mirror.length);
    for (var i=0; i<this.mirror.length; i++) {
      unscrambled[i] = this.mirror[pivots[i]];
    }
  }
  
  return unscrambled;
};

var exports = {};

/**
 * Solve A.x = b
 */
exports.solve = function solve(A, b) {
  return new Mat(A, b).gauss();
};

/**
 * return the identity matrix of size n
 */
function identity(n) {
  var id = new Array(n);
  for (var i=0; i<n; i++) {
    id[i] = new Array(n);
    for (var j=0; j<n; j++) {
      id[i][j] = (i === j) ? 1 : 0;
    }
  }
  return id;
}

/**
 * invert a matrix
 */
exports.invert = function invert(A) {
  return new Mat(A, identity(A.length)).gauss();
};

return exports;
})();

// Export as ES module
export default linearSolveModule;
export const solve = linearSolveModule.solve;
export const invert = linearSolveModule.invert;
