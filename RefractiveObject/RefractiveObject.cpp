//--------------------------------------------------------------------------------------
// File: BasicHLSL11.cpp
//
// This sample shows a simple example of the Microsoft Direct3D's High-Level 
// Shader Language (HLSL) using the Effect interface. 
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DXUT.h"
#include "DXUTcamera.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"
#include "SDKMesh.h"
#include "resource.h"
#include <xnamath.h>
#include <string>

#include "skybox11.h"
#include <D3DX11tex.h>
#include <D3DX11.h>
#include <D3DX11core.h>
#include <D3DX11async.h>

CSkybox11                   g_Skybox;
ID3D11Texture2D*            g_pTexRender11 = NULL;          // Render target texture for the skybox
ID3D11Texture2D*            g_pTexRenderMS11 = NULL;        // Render target texture for the skybox when multi sampling is on

typedef struct SimpleVertex
{
    XMFLOAT3 Pos;
    XMFLOAT2 Tex;
}node;

struct Vertex
{
    XMFLOAT3 Pos;
    XMFLOAT4 Color;
};

typedef struct R8G8B8
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
}color;

typedef struct v
{
    char voxel1;
    char voxel2;
}voxelSt;
//--------------
int mapSeq(float* in, float **out, int size, int length);
int marchPhoton(unsigned char* ocTree, unsigned char* nTree_c, float* direction, float* radiance,
    float* photondir, float* photonrad, float* photonpos, int exp, int num, float scale,
    float** Table, int tableSize, float tableScale, float tmpn);
int collectPhoton(unsigned char* ocTree, unsigned char* nTree_c, float* direction, float* radiance,
    float* photondir, float* photonrad, float* photonpos, int exp, int num, float scale,
    int **o_offset, int  **o_tableOffset, int **o_flag,
    float p1x, float p1y, float p1z, float p2x, float p2y, float p2z,
    float p3x, float p3y, float p3z, float p4x, float p4y, float p4z,
    float tmpn);
void gaussian2D(float* texture_d, float** textureout, int len, int kernelsize, float sigma, float scale);
int demapSeq(float** in);
int constructOctree(unsigned char *ri, int exp, unsigned char **out, float delta);

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
CDXUTDialogResourceManager  g_DialogResourceManager; // manager for shared resources of dialogs
CModelViewerCamera          g_Camera;               // A model viewing camera
//CFirstPersonCamera          g_Camera;               // A model viewing camera
CDXUTDirectionWidget        g_LightControl;
CD3DSettingsDlg             g_D3DSettingsDlg;       // Device settings dialog
CDXUTDialog                 g_HUD;                  // manages the 3D   
CDXUTDialog                 g_SampleUI;             // dialog for sample specific controls
D3DXMATRIXA16               g_mCenterMesh;
float                       g_fLightScale;
int                         g_nNumActiveLights;
int                         g_nActiveLight;
bool                        g_bShowHelp = false;    // If true, it renders the UI control text

// Direct3D9 resources
CDXUTTextHelper*            g_pTxtHelper = NULL;

CDXUTSDKMesh                g_Mesh11;
ID3D11Resource* tableTexture;
ID3D11InputLayout*          g_pVertexLayout11 = NULL;
ID3D11InputLayout*          photonVertexLayout = NULL;
ID3D11Buffer*               g_pVertexBuffer = NULL;
ID3D11Buffer*               g_pIndexBuffer = NULL;
ID3D11VertexShader*         g_pVertexShader = NULL;
ID3D11PixelShader*          g_pPixelShader = NULL;
ID3D11PixelShader*          g_pPixelShader1 = NULL;
ID3D11SamplerState*         g_pSamLinear = NULL;

ID3D11Buffer * imgVertexBuf = NULL;
ID3D11Buffer * imgIndexBuf = NULL;

ID3D11VertexShader*         photonVertexShader = NULL;
ID3D11PixelShader*          photonPixelShaderPos = NULL;
ID3D11PixelShader*          photonPixelShaderDir = NULL;

ID3D11ShaderResourceView*           g_pTextureRV = NULL;
ID3D11ShaderResourceView*           imgTextureRVBack = NULL;

ID3D11Texture2D *tmp;

// Setup the camera's view parameters
D3DXVECTOR3 vecEye(1.0f, 0.0f, 0.0f);
D3DXVECTOR3 vecAt(0.0f, 0.0f, 0.0f);
//最终成像所用
D3DXVECTOR3 vecEyeImg(100.0f, 0.0f, 0.0f);
D3DXVECTOR3 vecAtImg(0.0f, 0.0f, 0.0f);

D3DXVECTOR3 vecEyeView(180.0f, 460.0f, 161.0f);
//D3DXVECTOR3 vecEyeView(0.0f, 600.0f, 0.0f);
D3DXVECTOR3 vecAtView(0.0f, 0.0f, 0.0f);

float planeLen = 800.0f;

bool bRenderLight = false;
bool bRenderView = false;
bool bRenderShadow = false;

FLOAT fObjectRadius = 300.0f;
float farPlane = 9000.0f;
float nearPlane = 1.0f;
int meshWidth;
int meshHeight;
int meshDepth;
bool flagVoxel = false;
bool flagShowVoxel = false;
int windowWidth = 512;
int windowHeight = 512;
float meshScale = 1.0f;
float wrapRadius = 0.0f;
float w_long = 0.0f;
float h_long = 0.0f;
float d_long = 0.0f;
voxelSt voxel[128][128][128];
bool flagStoreArray = false;
bool flagOutput = false;
float x_min = 0;
float x_max = 0;

float y_min = 0;
float y_max = 0;

float z_min = 0;
float z_max = 0;
float N = 0.6;
bool marchBool = false;
float *direction = 0, *radiance = 0;
float *table=0;
float radScale = 0;

float pureRad = false;
struct CB_VS_PER_OBJECT
{
    D3DXMATRIX m_WorldViewProj;
    D3DXMATRIX m_World;
};
UINT                        g_iCBVSPerObjectBind = 0;

struct CB_PS_PER_OBJECT
{
    D3DXVECTOR4 m_vObjectColor;
};
UINT                        g_iCBPSPerObjectBind = 0;

struct CB_PS_PER_FRAME
{
    D3DXVECTOR4 m_vLightDirAmbient;
};
UINT                        g_iCBPSPerFrameBind = 1;

ID3D11Buffer*               g_pcbVSPerObject = NULL;
ID3D11Buffer*               g_pcbPSPerObject = NULL;
ID3D11Buffer*               g_pcbPSPerFrame = NULL;

//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN    1
#define IDC_TOGGLEREF           3
#define IDC_CHANGEDEVICE        4

//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext );
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext );
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext );
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext );
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );

extern bool CALLBACK IsD3D9DeviceAcceptable( D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, D3DFORMAT BackBufferFormat,
                                             bool bWindowed, void* pUserContext );
extern HRESULT CALLBACK OnD3D9CreateDevice( IDirect3DDevice9* pd3dDevice,
                                            const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
extern HRESULT CALLBACK OnD3D9ResetDevice( IDirect3DDevice9* pd3dDevice, const D3DSURFACE_DESC* pBackBufferSurfaceDesc,
                                           void* pUserContext );
extern void CALLBACK OnD3D9FrameRender( IDirect3DDevice9* pd3dDevice, double fTime, float fElapsedTime,
                                        void* pUserContext );
extern void CALLBACK OnD3D9LostDevice( void* pUserContext );
extern void CALLBACK OnD3D9DestroyDevice( void* pUserContext );

bool CALLBACK IsD3D11DeviceAcceptable(const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext );
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                      void* pUserContext );
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                          const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext );
void CALLBACK OnD3D11DestroyDevice( void* pUserContext );
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                  float fElapsedTime, void* pUserContext );

void InitApp();
void RenderText();


//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

    // DXUT will create and use the best device (either D3D9 or D3D11) 
    // that is available on the system depending on which D3D callbacks are set below

    // Set DXUT callbacks
    DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackKeyboard( OnKeyboard );
    DXUTSetCallbackFrameMove( OnFrameMove );


    DXUTSetCallbackD3D9DeviceAcceptable( IsD3D9DeviceAcceptable );
    DXUTSetCallbackD3D9DeviceCreated( OnD3D9CreateDevice );
    DXUTSetCallbackD3D9DeviceReset( OnD3D9ResetDevice );
    DXUTSetCallbackD3D9FrameRender( OnD3D9FrameRender );
    DXUTSetCallbackD3D9DeviceLost( OnD3D9LostDevice );
    DXUTSetCallbackD3D9DeviceDestroyed( OnD3D9DestroyDevice );


    DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
    DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
    DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
    DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );
    DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
    DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );

    InitApp();
    DXUTInit( true, true, NULL ); // Parse the command line, show msgboxes on error, no extra command line params
    DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
    DXUTCreateWindow( L"Advanced Computer Graphics" );
    DXUTCreateDevice(D3D_FEATURE_LEVEL_9_2, true, windowWidth, windowHeight);

    DXUTMainLoop(); // Enter into the DXUT render loop

    return DXUTGetExitCode();
}


//--------------------------------------------------------------------------------------
// Initialize the app 
//--------------------------------------------------------------------------------------
void InitApp()
{
    D3DXVECTOR3 vLightDir( -1, 1, -1 );
    D3DXVec3Normalize( &vLightDir, &vLightDir );
    g_LightControl.SetLightDirection( vLightDir );

    // Initialize dialogs
    g_D3DSettingsDlg.Init( &g_DialogResourceManager );
    g_HUD.Init( &g_DialogResourceManager );
    g_SampleUI.Init( &g_DialogResourceManager );

    g_HUD.SetCallback( OnGUIEvent ); int iY = 10;
    g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 0, iY, 170, 23 );
    g_HUD.AddButton( IDC_TOGGLEREF, L"Toggle REF (F3)", 0, iY += 26, 170, 23, VK_F3 );
    g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", 0, iY += 26, 170, 23, VK_F2 );

    g_SampleUI.SetCallback( OnGUIEvent ); iY = 10;
}


//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D11 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
    // Uncomment this to get debug information from D3D11
    //pDeviceSettings->d3d11.CreateFlags |= D3D11_CREATE_DEVICE_DEBUG;

    // For the first device created if its a REF device, optionally display a warning dialog box
    static bool s_bFirstTime = true;
    if( s_bFirstTime )
    {
        s_bFirstTime = false;
        if( ( DXUT_D3D11_DEVICE == pDeviceSettings->ver &&
              pDeviceSettings->d3d11.DriverType == D3D_DRIVER_TYPE_REFERENCE ) )
        {
            DXUTDisplaySwitchingToREFWarning( pDeviceSettings->ver );
        }
    }

    return true;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene.  This is called regardless of which D3D API is used
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
    // Update the camera's position based on user input 
    g_Camera.FrameMove( fElapsedTime );
}


//--------------------------------------------------------------------------------------
// Render the help and statistics text
//--------------------------------------------------------------------------------------
void RenderText()
{
    UINT nBackBufferHeight = ( DXUTIsAppRenderingWithD3D9() ) ? DXUTGetD3D9BackBufferSurfaceDesc()->Height :
            DXUTGetDXGIBackBufferSurfaceDesc()->Height;

    g_pTxtHelper->Begin();
    g_pTxtHelper->SetInsertionPos( 2, 0 );
    g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
    g_pTxtHelper->DrawTextLine( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) );
    g_pTxtHelper->DrawTextLine( DXUTGetDeviceStats() );

    // Draw help
    if( g_bShowHelp )
    {
        g_pTxtHelper->SetInsertionPos( 2, nBackBufferHeight - 20 * 6 );
        g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 0.75f, 0.0f, 1.0f ) );
        g_pTxtHelper->DrawTextLine( L"Controls:" );

        g_pTxtHelper->SetInsertionPos( 20, nBackBufferHeight - 20 * 5 );
        g_pTxtHelper->DrawTextLine( L"Rotate model: Left mouse button\n"
                                    L"Rotate light: Right mouse button\n"
                                    L"Rotate camera: Middle mouse button\n"
                                    L"Zoom camera: Mouse wheel scroll\n" );

        g_pTxtHelper->SetInsertionPos( 550, nBackBufferHeight - 20 * 5 );
        g_pTxtHelper->DrawTextLine( L"Hide help: F1\n"
                                    L"Quit: ESC\n" );
    }
    else
    {
        g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 1.0f, 1.0f ) );
        g_pTxtHelper->DrawTextLine( L"Press F1 for help" );
    }

    g_pTxtHelper->End();
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext )
{
    // Pass messages to dialog resource manager calls so GUI state is updated correctly
    *pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    // Pass messages to settings dialog if its active
    if( g_D3DSettingsDlg.IsActive() )
    {
        g_D3DSettingsDlg.MsgProc( hWnd, uMsg, wParam, lParam );
        return 0;
    }

    // Give the dialogs a chance to handle the message first
    *pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;
    *pbNoFurtherProcessing = g_SampleUI.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    g_LightControl.HandleMessages( hWnd, uMsg, wParam, lParam );

    // Pass all remaining windows messages to camera so it can respond to user input
    g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam );

    return 0;
}


//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
    float scale = 200.0f;
    if( bKeyDown )
    {
        switch( nChar )
        {
            case VK_F1:
                g_bShowHelp = !g_bShowHelp; break;
            case 'D':{
                bRenderView = false;
                vecEyeView.x += 0.1f * scale;
                g_Camera.SetViewParams(&vecEyeView, &vecAtView);
                break;
            }
            case 'A':{
                bRenderView = false;
                vecEyeView.x -= 0.1f * scale;
                g_Camera.SetViewParams(&vecEyeView, &vecAtView);
                         break;
            }
            case 'S':{
                bRenderView = false;
                vecEyeView.y -= 0.1f * scale;
                g_Camera.SetViewParams(&vecEyeView, &vecAtView);
                         break;
            }
            case 'W':{
                bRenderView = false;
                vecEyeView.y += 0.1f * scale;
                g_Camera.SetViewParams(&vecEyeView, &vecAtView);

                         break;
            }
            case 'Q':{
                bRenderView = false;
                vecEyeView.z += 0.1f * scale;
                g_Camera.SetViewParams(&vecEyeView, &vecAtView);
                         break;
            }
            case 'E':{
                bRenderView = false;
                vecEyeView.z -= 0.1f * scale;
                g_Camera.SetViewParams(&vecEyeView, &vecAtView);
                         break;
            }
            case 'O':{
                N = N - 0.1;
                marchBool = false;
                break;
            }
            case 'P':{
                N = N + 0.1;
                marchBool = false;
                break;
            }
            case 'K':{
                radScale = radScale + 1;
                break;
            }
            case 'L':{
                radScale = radScale - 1;
                break;
            }
            case 'I':{
                pureRad = !pureRad;
                break;
            }
            case 'Y':{
                bRenderShadow = !bRenderShadow;
                break;
            }
        }
    }
}


//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
    switch( nControlID )
    {
        case IDC_TOGGLEFULLSCREEN:
            DXUTToggleFullScreen(); break;
        case IDC_TOGGLEREF:
            DXUTToggleREF(); break;
        case IDC_CHANGEDEVICE:
            g_D3DSettingsDlg.SetActive( !g_D3DSettingsDlg.IsActive() ); break;
    }

}


//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
    return true;
}

//--------------------------------------------------------------------------------------
// Find and compile the specified shader
//--------------------------------------------------------------------------------------
HRESULT CompileShaderFromFile( WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3DBlob** ppBlobOut )
{
    HRESULT hr = S_OK;

    // find the file
    WCHAR str[MAX_PATH];
    V_RETURN( DXUTFindDXSDKMediaFileCch( str, MAX_PATH, szFileName ) );

    DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
    // Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release configuration of this program.
    dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

    ID3DBlob* pErrorBlob;
    hr = D3DX11CompileFromFile( str, NULL, NULL, szEntryPoint, szShaderModel, 
        dwShaderFlags, 0, NULL, ppBlobOut, &pErrorBlob, NULL );
    if( FAILED(hr) )
    {
        if( pErrorBlob != NULL )
            OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
        SAFE_RELEASE( pErrorBlob );
        return hr;
    }
    SAFE_RELEASE( pErrorBlob );

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                      void* pUserContext )
{
    HRESULT hr;

    ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();
    V_RETURN( g_DialogResourceManager.OnD3D11CreateDevice( pd3dDevice, pd3dImmediateContext ) );
    V_RETURN( g_D3DSettingsDlg.OnD3D11CreateDevice( pd3dDevice ) );
    g_pTxtHelper = new CDXUTTextHelper( pd3dDevice, pd3dImmediateContext, &g_DialogResourceManager, 15 );

    D3DXVECTOR3 vCenter( 0.0f, 0.0f, 0.0f );
    //FLOAT fObjectRadius = 378.15607f;

    D3DXMatrixTranslation( &g_mCenterMesh, -vCenter.x, -vCenter.y, -vCenter.z );

    // Compile the shaders using the lowest possible profile for broadest feature level support
    ID3DBlob* pVertexShaderBuffer = NULL;
    V_RETURN( CompileShaderFromFile( L"BasicHLSL11_VS.hlsl", "VSMain", "vs_4_0_level_9_1", &pVertexShaderBuffer ) );

    ID3DBlob* pPixelShaderBuffer = NULL;
    V_RETURN( CompileShaderFromFile( L"BasicHLSL11_PS.hlsl", "PSMain", "ps_4_0_level_9_1", &pPixelShaderBuffer ) );

    ID3DBlob* photonVertexShaderBuffer = NULL;
    V_RETURN(CompileShaderFromFile(L"photonVS.hlsl", "VSMain", "vs_4_0_level_9_1", &photonVertexShaderBuffer));

    ID3DBlob* photonPixelShaderBufferPos = NULL;
    V_RETURN(CompileShaderFromFile(L"photonPSPos.hlsl", "PSMain", "ps_4_0_level_9_1", &photonPixelShaderBufferPos));

    ID3DBlob* photonPixelShaderBufferDir = NULL;
    V_RETURN(CompileShaderFromFile(L"photonPSDir.hlsl", "PSMain", "ps_4_0_level_9_1", &photonPixelShaderBufferDir));

    // Create the shaders
    V_RETURN( pd3dDevice->CreateVertexShader( pVertexShaderBuffer->GetBufferPointer(),
                                              pVertexShaderBuffer->GetBufferSize(), NULL, &g_pVertexShader ) );
    DXUT_SetDebugName( g_pVertexShader, "VSMain" );

    V_RETURN( pd3dDevice->CreatePixelShader( pPixelShaderBuffer->GetBufferPointer(),
                                             pPixelShaderBuffer->GetBufferSize(), NULL, &g_pPixelShader ) );
    DXUT_SetDebugName( g_pPixelShader, "PSMain" );

    //光子的3个shader
    V_RETURN(pd3dDevice->CreateVertexShader(photonVertexShaderBuffer->GetBufferPointer(),
        photonVertexShaderBuffer->GetBufferSize(), NULL, &photonVertexShader));
    DXUT_SetDebugName(photonVertexShader, "VSMain");

    V_RETURN(pd3dDevice->CreatePixelShader(photonPixelShaderBufferPos->GetBufferPointer(),
        photonPixelShaderBufferPos->GetBufferSize(), NULL, &photonPixelShaderPos));
    DXUT_SetDebugName(photonPixelShaderPos, "PSMain");

    V_RETURN(pd3dDevice->CreatePixelShader(photonPixelShaderBufferDir->GetBufferPointer(),
        photonPixelShaderBufferDir->GetBufferSize(), NULL, &photonPixelShaderDir));
    DXUT_SetDebugName(photonPixelShaderDir, "PSMain");

    // Create our vertex input layout
    const D3D11_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD",  0, DXGI_FORMAT_R32G32_FLOAT,    0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    V_RETURN( pd3dDevice->CreateInputLayout( layout, ARRAYSIZE( layout ), pVertexShaderBuffer->GetBufferPointer(),
                                             pVertexShaderBuffer->GetBufferSize(), &g_pVertexLayout11 ) );

    DXUT_SetDebugName( g_pVertexLayout11, "Primary" );

    D3D11_INPUT_ELEMENT_DESC vertexDesc[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };

    // Create the input layout
    V_RETURN(pd3dDevice->CreateInputLayout(vertexDesc, ARRAYSIZE(vertexDesc), photonVertexShaderBuffer->GetBufferPointer(),
        photonVertexShaderBuffer->GetBufferSize(), &photonVertexLayout));

    DXUT_SetDebugName(photonVertexLayout, "Primary");


    SAFE_RELEASE( pVertexShaderBuffer );
    SAFE_RELEASE( pPixelShaderBuffer );
    SAFE_RELEASE(photonVertexShaderBuffer);
    SAFE_RELEASE(photonPixelShaderBufferPos);
    SAFE_RELEASE(photonPixelShaderBufferDir);

    // Load the mesh
    //V_RETURN( g_Mesh11.Create( pd3dDevice, L"tiny\\tiny.sdkmesh", true ) );
    //V_RETURN(g_Mesh11.Create(pd3dDevice, L"tiny\\h.sdkmesh", true));
    V_RETURN(g_Mesh11.Create(pd3dDevice, L"k.sdkmesh", true));
    //========== ===========
    //为了最终成像
    SimpleVertex imgVertices[] =
    {
        { XMFLOAT3(-256, 0.0, 256.0),  XMFLOAT2(0.0f, 0.0f) },
        { XMFLOAT3(256.0, 0.0, 256.0), XMFLOAT2(1.0f, 0.0f) },
        { XMFLOAT3(256.0, 0.0, -256.0),  XMFLOAT2(1.0f, 1.0f) },
        { XMFLOAT3(-256.0, 0.0, -256.0), XMFLOAT2(0.0f, 1.0f) }
    };

    D3D11_BUFFER_DESC vbdi;
    vbdi.Usage = D3D11_USAGE_IMMUTABLE;
    vbdi.ByteWidth = sizeof(SimpleVertex) * 4;
    vbdi.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vbdi.CPUAccessFlags = 0;
    vbdi.MiscFlags = 0;
    vbdi.StructureByteStride = 0;
    D3D11_SUBRESOURCE_DATA vinitDatai;
    vinitDatai.pSysMem = imgVertices;
    V_RETURN(pd3dDevice->CreateBuffer(&vbdi, &vinitDatai, &imgVertexBuf));

    UINT imgIndices[] = {
        // front face
        0, 1, 3,
        1, 2, 3
    };

    D3D11_BUFFER_DESC ibdi;
    ibdi.Usage = D3D11_USAGE_IMMUTABLE;
    ibdi.ByteWidth = sizeof(UINT) * 6;
    ibdi.BindFlags = D3D11_BIND_INDEX_BUFFER;
    ibdi.CPUAccessFlags = 0;
    ibdi.MiscFlags = 0;
    ibdi.StructureByteStride = 0;
    D3D11_SUBRESOURCE_DATA iinitDatai;
    iinitDatai.pSysMem = imgIndices;
    V_RETURN(pd3dDevice->CreateBuffer(&ibdi, &iinitDatai, &imgIndexBuf));

    D3D11_SAMPLER_DESC sampDesc;
    ZeroMemory(&sampDesc, sizeof(sampDesc));
    sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampDesc.MinLOD = 0;
    sampDesc.MaxLOD = D3D11_FLOAT32_MAX;
    hr = pd3dDevice->CreateSamplerState(&sampDesc, &g_pSamLinear);

    if (FAILED(hr))
        return hr;

     /*Load the Texture*/
    hr = D3DX11CreateShaderResourceViewFromFile(pd3dDevice, L"seafloor.dds", NULL, NULL, &g_pTextureRV, NULL);
    if (FAILED(hr))
        return hr;

    // Setup constant buffers
    D3D11_BUFFER_DESC Desc;
    Desc.Usage = D3D11_USAGE_DYNAMIC;
    Desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    Desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    Desc.MiscFlags = 0;
    Desc.ByteWidth = sizeof( CB_VS_PER_OBJECT );

    V_RETURN( pd3dDevice->CreateBuffer( &Desc, NULL, &g_pcbVSPerObject ) );
    DXUT_SetDebugName( g_pcbVSPerObject, "CB_VS_PER_OBJECT" );

    Desc.ByteWidth = sizeof( CB_PS_PER_OBJECT );
    V_RETURN( pd3dDevice->CreateBuffer( &Desc, NULL, &g_pcbPSPerObject ) );
    DXUT_SetDebugName( g_pcbPSPerObject, "CB_PS_PER_OBJECT" );

    Desc.ByteWidth = sizeof( CB_PS_PER_FRAME );
    V_RETURN( pd3dDevice->CreateBuffer( &Desc, NULL, &g_pcbPSPerFrame ) );
    DXUT_SetDebugName( g_pcbPSPerFrame, "CB_PS_PER_FRAME" );

    UINT Strides[1];
    UINT Offsets[1];

    Strides[0] = sizeof(SimpleVertex);
    Offsets[0] = 0;
    ID3D11Buffer* pVB[1];
    pVB[0] = g_Mesh11.GetVB11(0, 0);
    Strides[0] = (UINT)g_Mesh11.GetVertexStride(0, 0);
    Offsets[0] = 0;
    int size = g_Mesh11.GetNumVertices(0, 0);

    ID3D11Buffer * cpubuffer = NULL;
    D3D11_BUFFER_DESC abufferDesc;
    ZeroMemory(&abufferDesc, sizeof(abufferDesc));
    pVB[0]->GetDesc(&abufferDesc);
    abufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    abufferDesc.Usage = D3D11_USAGE_STAGING;
    abufferDesc.BindFlags = 0;
    abufferDesc.MiscFlags = 0;

    //ID3D11Buffer* pDebugBuffer = NULL;
    pd3dDevice->CreateBuffer(&abufferDesc, NULL, &cpubuffer);
    pd3dImmediateContext->CopyResource(cpubuffer, pVB[0]);
    D3D11_MAPPED_SUBRESOURCE resultResources;
    ZeroMemory(&resultResources, sizeof(D3D11_MAPPED_SUBRESOURCE));
    pd3dImmediateContext->Map(cpubuffer, 0, D3D11_MAP_READ, 0, &resultResources);
    node * ptn = (node *)(resultResources.pData);

    node* m_pGPUTestResult = new node[size];
    for (int i = 0; i<size; i++)
    {
        m_pGPUTestResult[i].Pos.x = ptn[i].Pos.x;
        m_pGPUTestResult[i].Pos.y = ptn[i].Pos.y;
        m_pGPUTestResult[i].Pos.y = ptn[i].Pos.z;
        if (ptn[i].Pos.x > x_max)
        {
            x_max = ptn[i].Pos.x;
        }
        if (ptn[i].Pos.x < x_min)
        {
            x_min = ptn[i].Pos.x;
        }
        if (ptn[i].Pos.y > y_max)
        {
            y_max = ptn[i].Pos.y;
        }
        if (ptn[i].Pos.y < y_min)
        {
            y_min = ptn[i].Pos.y;
        }
        if (ptn[i].Pos.z > z_max)
        {
            z_max = ptn[i].Pos.z;
        }
        if (ptn[i].Pos.z < z_min)
        {
            z_min = ptn[i].Pos.z;
        }
    }

    x_min -= 20;
    y_min -= 20;
    z_min -= 20;
    x_max += 20;
    y_max += 20;
    z_max += 20;

    meshWidth = ((int)(x_max - x_min) / 100);
    if ((int)(x_max - x_min) % 100 != 0)
        meshWidth++;
    meshWidth *= 100;
    meshHeight = ((int)(y_max - y_min) / 100);
    if ((int)(y_max - y_min) % 100 != 0)
        meshHeight++;
    meshHeight *= 100;
    meshDepth = ((int)(z_max - z_min) / 100);
    if ((int)(z_max - z_min) % 100 != 0)
        meshDepth++;
    meshDepth *= 100;

    w_long = max(fabs(x_max), fabs(x_min));
    h_long = max(fabs(z_max), fabs(z_min));
    d_long = max(fabs(y_max), fabs(y_min));

    float max = max(w_long, max(h_long, d_long));

    meshHeight = windowHeight;
    pd3dImmediateContext->Unmap(cpubuffer, 0);
    cpubuffer->Release();
    
    V_RETURN(g_DialogResourceManager.OnD3D11CreateDevice(pd3dDevice, pd3dImmediateContext));
    V_RETURN(g_D3DSettingsDlg.OnD3D11CreateDevice(pd3dDevice));
    g_pTxtHelper = new CDXUTTextHelper(pd3dDevice, pd3dImmediateContext, &g_DialogResourceManager, 15);

    WCHAR strPath[MAX_PATH];
    V_RETURN(DXUTFindDXSDKMediaFileCch(strPath, MAX_PATH, L"stpeters_cross.dds"));

    ID3D11Texture2D* pCubeTexture = NULL;
    ID3D11ShaderResourceView* pCubeRV = NULL;
    UINT SupportCaps = 0;

    pd3dDevice->CheckFormatSupport(DXGI_FORMAT_R32G32B32A32_FLOAT, &SupportCaps);
    if (SupportCaps & D3D11_FORMAT_SUPPORT_TEXTURECUBE &&
        SupportCaps & D3D11_FORMAT_SUPPORT_RENDER_TARGET &&
        SupportCaps & D3D11_FORMAT_SUPPORT_TEXTURE2D)
    {
        D3DX11_IMAGE_LOAD_INFO LoadInfo;
        LoadInfo.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        V_RETURN(D3DX11CreateShaderResourceViewFromFile(pd3dDevice, strPath, &LoadInfo, NULL, &pCubeRV, NULL));
        DXUT_SetDebugName(pCubeRV, "uffizi_cross.dds");
        pCubeRV->GetResource((ID3D11Resource**)&pCubeTexture);
        DXUT_SetDebugName(pCubeTexture, "uffizi_cross.dds");
        V_RETURN(g_Skybox.OnD3D11CreateDevice(pd3dDevice, 50, pCubeTexture, pCubeRV));
    }
    else
        return E_FAIL;

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                          const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;
    g_Skybox.OnD3D11ResizedSwapChain(pBackBufferSurfaceDesc);
    V_RETURN( g_DialogResourceManager.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );
    V_RETURN( g_D3DSettingsDlg.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );

    // Setup the camera's projection parameters
    float fAspectRatio = pBackBufferSurfaceDesc->Width / ( FLOAT )pBackBufferSurfaceDesc->Height;
    g_Camera.SetProjParams( D3DX_PI / 4, fAspectRatio, /*50.0f*/nearPlane, /*4000.0f*/farPlane );
    g_Camera.SetWindow( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );
    g_Camera.SetButtonMasks( MOUSE_MIDDLE_BUTTON, MOUSE_WHEEL, MOUSE_LEFT_BUTTON );

    return S_OK;
}


bool toBackTextureDown = false;
ID3D11RenderTargetView*     resTVBack;
ID3D11Texture2D*            resTexBack = NULL;
ID3D11Texture2D*            resTexBackCopy = NULL;
//--------------------------------------------------------------------------------------
// Get the background image from the skybox 
//--------------------------------------------------------------------------------------
void renderToBackTexture(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext){
    if (toBackTextureDown)
        return;
    toBackTextureDown = true;
    UINT Strides[1];
    ID3D11Buffer* pVB[1];
    UINT Offsets[1];
    //pVB[0] = g_Mesh11.GetVB11(0, 0);
    UINT stride = sizeof(SimpleVertex);
    UINT offset = 0;

    //the description of the texture 
    D3D11_TEXTURE2D_DESC resTexDesc;
    ZeroMemory(&resTexDesc, sizeof(resTexDesc));
    IDXGISwapChain*    pSwapChain = DXUTGetDXGISwapChain();
    ID3D11Texture2D *backBuffer(NULL);

    pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
    backBuffer->GetDesc(&resTexDesc);
    resTexDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
    backBuffer->Release();

    //the texture2D binded with the render target view
    HRESULT hr = pd3dDevice->CreateTexture2D(&resTexDesc, NULL, &resTexBack);
    if (FAILED(hr)){
        char tmpstr[128];
        wsprintfA(tmpstr, "Error code : %lX", hr);
        OutputDebugStringA(tmpstr);
        //ErrorMessage(hr);
        return;
    }

    //new render target view description
    D3D11_RENDER_TARGET_VIEW_DESC resTVDesc;
    ZeroMemory(&resTVBack, sizeof(resTVBack));
    ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
    pRTV->GetDesc(&resTVDesc);
    resTVDesc.Format = resTexDesc.Format;
    resTVDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    resTVDesc.Texture2D.MipSlice = 0;

    //create the targetview
    hr = pd3dDevice->CreateRenderTargetView(resTexBack, &resTVDesc, &resTVBack);
    if (FAILED(hr)){
        char tmpstr[128];
        wsprintfA(tmpstr, "Error code : %lX", hr);
        OutputDebugStringA(tmpstr);
        return;
    }

    //clear
    ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
    pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
    float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    pd3dImmediateContext->ClearRenderTargetView(resTVBack, ClearColor);
    pd3dImmediateContext->OMSetRenderTargets(1, &resTVBack, pDSV);

    D3DXMATRIX mWorldViewProjection;
    D3DXMATRIX mWorld;
    D3DXMATRIX mView;
    D3DXMATRIX mProj;

    D3DXVECTOR3 vecEyeN(vecEyeView.x, vecEyeView.z, vecEyeView.y);
    D3DXVECTOR3 vecAtN(vecAt.x, vecAt.z, vecAt.y);

    g_Camera.SetViewParams(&vecEyeN, &vecAtN);
    mWorld = *g_Camera.GetWorldMatrix();
    mProj = *g_Camera.GetProjMatrix();
    mView = *g_Camera.GetViewMatrix();
    mWorldViewProjection = mWorld * mView * mProj;

    g_Skybox.D3D11Render(&mWorldViewProjection, pd3dImmediateContext);

    //copy the texture rendered
    D3D11_TEXTURE2D_DESC copyDesc;
    resTexBack->GetDesc(&copyDesc);
    copyDesc.BindFlags = 0;
    copyDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    copyDesc.Usage = D3D11_USAGE_STAGING;
    //copyDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    copyDesc.MiscFlags = 0;
    hr = pd3dDevice->CreateTexture2D(&copyDesc, NULL, &resTexBackCopy);
    if (FAILED(hr))
        return;

    pd3dImmediateContext->CopyResource(resTexBackCopy, resTexBack);
    pd3dImmediateContext->OMSetRenderTargets(1, &pRTV, pDSV);

}

//light photon travel
ID3D11Buffer * lightVertexBuf = NULL;
ID3D11Buffer * lightIndexBuf = NULL;

ID3D11RenderTargetView*     lightTV;
ID3D11Texture2D*            lightTex = NULL;
ID3D11Texture2D*            lightTexCopy = NULL;

float posLight[262144][3] = { 0 };
float dirLight[262144][3] = { 0 };

D3DXVECTOR3 vecEyeLight(0.0f, 0.0f, 600.0f);
D3DXVECTOR3 vecAtLight(0.0f, 0.0f, 0.0f);
int photonNumLight = 0;

void renderLight(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext){
    HRESULT hr;
    //========== ===========
    Vertex lightVertices[] =
    {
        { XMFLOAT3(x_min, y_min, z_min), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_min, y_max, z_min), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_max, y_max, z_min), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_max, y_min, z_min), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_min, y_min, z_max), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_min, y_max, z_max), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_max, y_max, z_max), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_max, y_min, z_max), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) }
    };

    D3D11_BUFFER_DESC vbdLight;
    vbdLight.Usage = D3D11_USAGE_IMMUTABLE;
    vbdLight.ByteWidth = sizeof(Vertex) * 8;
    vbdLight.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vbdLight.CPUAccessFlags = 0;
    vbdLight.MiscFlags = 0;
    vbdLight.StructureByteStride = 0;
    D3D11_SUBRESOURCE_DATA vinitDataLight;
    vinitDataLight.pSysMem = lightVertices;
    hr = pd3dDevice->CreateBuffer(&vbdLight, &vinitDataLight, &lightVertexBuf);
    if (FAILED(hr)){
        char tmpstr[128];
        wsprintfA(tmpstr, "Error code : %lX", hr);
        OutputDebugStringA(tmpstr);
        //ErrorMessage(hr);
        return;
    }

    UINT lightIndices[] = {
        // front face
        0, 1, 2,
        0, 2, 3,

        // back face
        4, 6, 5,
        4, 7, 6,

        // left face
        4, 5, 1,
        4, 1, 0,

        // right face
        3, 2, 6,
        3, 6, 7,

        // top face
        1, 5, 6,
        1, 6, 2,

        // bottom face
        4, 0, 3,
        4, 3, 7
    };

    D3D11_BUFFER_DESC ibdLight;
    ibdLight.Usage = D3D11_USAGE_IMMUTABLE;
    ibdLight.ByteWidth = sizeof(UINT) * 36;
    ibdLight.BindFlags = D3D11_BIND_INDEX_BUFFER;
    ibdLight.CPUAccessFlags = 0;
    ibdLight.MiscFlags = 0;
    ibdLight.StructureByteStride = 0;
    D3D11_SUBRESOURCE_DATA iinitDataLight;
    iinitDataLight.pSysMem = lightIndices;
    hr = pd3dDevice->CreateBuffer(&ibdLight, &iinitDataLight, &lightIndexBuf);
    if (FAILED(hr)){
        char tmpstr[128];
        wsprintfA(tmpstr, "Error code : %lX", hr);
        OutputDebugStringA(tmpstr);
        //ErrorMessage(hr);
        return;
    }

    UINT Strides[1];
    UINT Offsets[1];
    UINT stride = sizeof(Vertex);
    UINT offset = 0;

    //创建target对应texture的描述
    D3D11_TEXTURE2D_DESC resTexDesc;
    ZeroMemory(&resTexDesc, sizeof(resTexDesc));
    IDXGISwapChain*    pSwapChain = DXUTGetDXGISwapChain();
    ID3D11Texture2D *backBuffer(NULL);
    pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
    backBuffer->GetDesc(&resTexDesc);
    resTexDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    resTexDesc.BindFlags = 0;
    resTexDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
    backBuffer->Release();

    //target对应的texture2D
    hr = pd3dDevice->CreateTexture2D(&resTexDesc, NULL, &lightTex);
    if (FAILED(hr)){
        char tmpstr[128];
        wsprintfA(tmpstr, "Error code : %lX", hr);
        OutputDebugStringA(tmpstr);
        //ErrorMessage(hr);
        return;
    }

    //new render target view
    D3D11_RENDER_TARGET_VIEW_DESC resTVDesc;
    ZeroMemory(&resTVDesc, sizeof(resTVDesc));
    ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
    pRTV->GetDesc(&resTVDesc);
    resTVDesc.Format = resTexDesc.Format;
    resTVDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    resTVDesc.Texture2D.MipSlice = 0;

    //创建 targetview
    hr = pd3dDevice->CreateRenderTargetView(lightTex, &resTVDesc, &lightTV);
    if (FAILED(hr)){
        char tmpstr[128];
        wsprintfA(tmpstr, "Error code : %lX", hr);
        OutputDebugStringA(tmpstr);
        return;
    }

    //clear
    ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
    pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
    float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    pd3dImmediateContext->ClearRenderTargetView(lightTV, ClearColor);
    pd3dImmediateContext->OMSetRenderTargets(1, &lightTV, pDSV);

    D3DXMATRIX mWorldViewProjection;
    D3DXVECTOR3 vLightDir;
    D3DXMATRIX mWorld;
    D3DXMATRIX mView;
    D3DXMATRIX mProj;
    D3DXMATRIX mTmp;

    //更新常数buffer
    // Per frame cb update
    D3D11_MAPPED_SUBRESOURCE MappedResource;
    V(pd3dImmediateContext->Map(g_pcbPSPerFrame, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource));
    CB_PS_PER_FRAME* pPerFrame = (CB_PS_PER_FRAME*)MappedResource.pData;
    float fAmbient = 0.1f;
    pPerFrame->m_vLightDirAmbient = D3DXVECTOR4(vecEyeLight.x, vecEyeLight.y, vecEyeLight.z, fAmbient);
    pd3dImmediateContext->Unmap(g_pcbPSPerFrame, 0);
    pd3dImmediateContext->PSSetConstantBuffers(g_iCBPSPerFrameBind, 1, &g_pcbPSPerFrame);

    pd3dImmediateContext->IASetInputLayout(photonVertexLayout);
    pd3dImmediateContext->IASetVertexBuffers(0, 1, &lightVertexBuf, &stride, &offset);
    pd3dImmediateContext->IASetIndexBuffer(lightIndexBuf, DXGI_FORMAT_R32_UINT, 0);

    pd3dImmediateContext->VSSetShader(photonVertexShader, NULL, 0);
    pd3dImmediateContext->PSSetShader(photonPixelShaderPos, NULL, 0);

    // Set the per object constant data
    D3DXMATRIX m_orthoMatrix;
    XMMatrixScaling(0.5f, 0.5f, 0.5f);
    D3DXMatrixOrthoLH(&m_orthoMatrix, (float)windowWidth, (float)windowHeight, /*50.0f*/nearPlane, farPlane/*4000.0f*/);
    D3DXMATRIX scale;
    //D3DXMatrixScaling(&scale, 0.5f, 0.5f, 0.5f);
    D3DXMatrixScaling(&scale, meshScale, meshScale, meshScale);

    D3DXMATRIX dView;
    D3DXVECTOR3 dEye(vecEyeLight.x, vecEyeLight.y, vecEyeLight.z);
    D3DXVECTOR3 dAt(0.0f, 0.0f, 0.0f);
    D3DXVECTOR3 dUp(0.5f, 0.5f, 1.0f);
    D3DXMatrixLookAtLH(&dView, &dEye, &dAt, &dUp);
    D3DXMATRIX dWorld;
    D3DXMatrixIdentity(&dWorld);
    D3DXMATRIX dProj;
    D3DXMatrixPerspectiveFovLH(&dProj, D3DX_PI / 4, 1.0f, 1.0f, 6000.0f);

    mWorldViewProjection = dWorld * dView * dProj;

    // VS Per object
    V(pd3dImmediateContext->Map(g_pcbVSPerObject, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource));
    CB_VS_PER_OBJECT* pVSPerObject = (CB_VS_PER_OBJECT*)MappedResource.pData;
    D3DXMatrixTranspose(&pVSPerObject->m_WorldViewProj, &mWorldViewProjection);
    D3DXMatrixTranspose(&pVSPerObject->m_World, &mWorld);
    pd3dImmediateContext->Unmap(g_pcbVSPerObject, 0);
    pd3dImmediateContext->VSSetConstantBuffers(g_iCBVSPerObjectBind, 1, &g_pcbVSPerObject);

    // PS Per object
    V(pd3dImmediateContext->Map(g_pcbPSPerObject, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource));
    CB_PS_PER_OBJECT* pPSPerObject = (CB_PS_PER_OBJECT*)MappedResource.pData;
    pPSPerObject->m_vObjectColor = D3DXVECTOR4(1, 1, 1, 1);
    pd3dImmediateContext->Unmap(g_pcbPSPerObject, 0);
    pd3dImmediateContext->PSSetConstantBuffers(g_iCBPSPerObjectBind, 1, &g_pcbPSPerObject);

    //Render
    D3D11_PRIMITIVE_TOPOLOGY PrimType;
    PrimType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    pd3dImmediateContext->IASetPrimitiveTopology(PrimType);
    pd3dImmediateContext->DrawIndexed(36, 0, 0);

    //将target复制出来
    D3D11_TEXTURE2D_DESC copyDesc;
    lightTex->GetDesc(&copyDesc);
    copyDesc.BindFlags = 0;
    copyDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    copyDesc.Usage = D3D11_USAGE_STAGING;
    copyDesc.BindFlags = 0;
    copyDesc.MiscFlags = 0;
    hr = pd3dDevice->CreateTexture2D(&copyDesc, NULL, &lightTexCopy);
    if (FAILED(hr))
        return;

    pd3dImmediateContext->CopyResource(lightTexCopy, lightTex);
    D3D11_MAPPED_SUBRESOURCE frameResources;
    ZeroMemory(&frameResources, sizeof(D3D11_MAPPED_SUBRESOURCE));
    pd3dImmediateContext->Map(lightTexCopy, 0, D3D11_MAP_READ, 0, &frameResources);
    XMFLOAT4 * ptr = (XMFLOAT4*)(frameResources.pData);
    int count = 0;

    for (size_t i = 0; i < 512; i++)
    {
        for (size_t j = 0; j < 512; j++)
        {
            if (ptr[i * 512 + j].w > 0.5){
                posLight[count][0] = ptr[i * 512 + j].x;
                posLight[count][1] = ptr[i * 512 + j].y;
                posLight[count][2] = ptr[i * 512 + j].z;
                count++;
            }
        }
    }
    photonNumLight = count;

    pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);

    pd3dImmediateContext->ClearRenderTargetView(lightTV, ClearColor);
    pd3dImmediateContext->OMSetRenderTargets(1, &lightTV, pDSV);

    pd3dImmediateContext->VSSetShader(photonVertexShader, NULL, 0);
    pd3dImmediateContext->PSSetShader(photonPixelShaderDir, NULL, 0);


    D3DXMatrixOrthoLH(&m_orthoMatrix, (float)windowWidth, (float)windowHeight, /*50.0f*/nearPlane, farPlane/*4000.0f*/);
    D3DXMatrixScaling(&scale, meshScale, meshScale, meshScale);

    //Render
    //D3D11_PRIMITIVE_TOPOLOGY PrimType;
    PrimType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    pd3dImmediateContext->IASetPrimitiveTopology(PrimType);
    pd3dImmediateContext->DrawIndexed(36, 0, 0);

    //将target复制出来
    //D3D11_TEXTURE2D_DESC copyDesc;
    lightTex->GetDesc(&copyDesc);
    copyDesc.BindFlags = 0;
    copyDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    copyDesc.Usage = D3D11_USAGE_STAGING;
    copyDesc.BindFlags = 0;
    copyDesc.MiscFlags = 0;
    hr = pd3dDevice->CreateTexture2D(&copyDesc, NULL, &lightTexCopy);
    if (FAILED(hr))
        return;

    pd3dImmediateContext->CopyResource(lightTexCopy, lightTex);
    //D3D11_MAPPED_SUBRESOURCE frameResources;
    ZeroMemory(&frameResources, sizeof(D3D11_MAPPED_SUBRESOURCE));
    pd3dImmediateContext->Map(lightTexCopy, 0, D3D11_MAP_READ, 0, &frameResources);
    ptr = (XMFLOAT4*)(frameResources.pData);
    count = 0;

    for (size_t i = 0; i < 512; i++)
    {
        for (size_t j = 0; j < 512; j++)
        {
            if (ptr[i * 512 + j].w > 0.5){
                dirLight[count][0] = ptr[i * 512 + j].x;
                dirLight[count][1] = ptr[i * 512 + j].y;
                dirLight[count][2] = ptr[i * 512 + j].z;
                float mul = dirLight[count][0] * dirLight[count][0] + dirLight[count][1] * dirLight[count][1] + dirLight[count][2] * dirLight[count][2];
                mul = sqrt(mul);
                dirLight[count][0] = ptr[i * 512 + j].x / mul;
                dirLight[count][1] = ptr[i * 512 + j].y / mul;
                dirLight[count][2] = ptr[i * 512 + j].z / mul;
                count++;
            }

        }
    }
    pd3dImmediateContext->OMSetRenderTargets(1, &pRTV, pDSV);

}


//View pass
ID3D11Buffer * viewVertexBuf = NULL;
ID3D11Buffer * viewIndexBuf = NULL;

ID3D11RenderTargetView*     viewTV;
ID3D11Texture2D*            viewTex = NULL;
ID3D11Texture2D*            viewTexCopy = NULL;

float posView[262144][3] = { 0 };
float dirView[262144][3] = { 0 };

void renderView(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext){
    
    HRESULT hr;
    float dx = vecAtView.x - vecEyeView.x;
    float dy = vecAtView.y - vecEyeView.y;
    float dz = vecAtView.z - vecEyeView.z;

    float muln = dx * dx + dy * dy + dz * dz;
    muln = sqrt(muln);

    dx /= muln;
    dy /= muln;
    dz /= muln;

    float centerX = vecEyeView.x + planeLen * dx;
    float centerY = vecEyeView.y + planeLen * dy;
    float centerZ = vecEyeView.z + planeLen * dz;

    float a;
    float b;
    if (abs(dy - 0.0) < 0.00000001){
        a = 0;
        b = 1.0;
    }
    else
    {
        a = 1.0f;
        b = -dx / dy;
    }

    muln = a * a + b * b;
    muln = sqrt(muln);

    a /= muln;
    b /= muln;

    float u = -dz * b;
    float v = dz * a;
    float w = dx * b - dy * a;

    muln = u*u + v*v + w*w;
    muln = sqrt(muln);

    u /= muln;
    v /= muln;
    w /= muln;

    //========== ===========
    Vertex viewVertices[] =
    {
        { XMFLOAT3(x_min, y_min, z_min), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_min, y_max, z_min), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_max, y_max, z_min), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_max, y_min, z_min), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_min, y_min, z_max), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_min, y_max, z_max), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_max, y_max, z_max), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(x_max, y_min, z_max), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(centerX - a * planeLen * 0.4142 + u * planeLen * 0.4142, centerY - planeLen * 0.4142 * b + planeLen * 0.4142 * v, centerZ + planeLen * 0.4142 * w), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(centerX + a * planeLen * 0.4142 + u * planeLen * 0.4142, centerY + planeLen * 0.4142 * b + planeLen * 0.4142 * v, centerZ + planeLen * 0.4142 * w), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(centerX + a * planeLen * 0.4142 - u * planeLen * 0.4142, centerY + planeLen * 0.4142 * b - planeLen * 0.4142 * v, centerZ - planeLen * 0.4142 * w), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) },
        { XMFLOAT3(centerX - a * planeLen * 0.4142 - u * planeLen * 0.4142, centerY - planeLen * 0.4142 * b - planeLen * 0.4142 * v, centerZ - planeLen * 0.4142 * w), XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f) }

    };

    D3D11_BUFFER_DESC vbdView;
    vbdView.Usage = D3D11_USAGE_IMMUTABLE;
    vbdView.ByteWidth = sizeof(Vertex) * 12;
    vbdView.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vbdView.CPUAccessFlags = 0;
    vbdView.MiscFlags = 0;
    vbdView.StructureByteStride = 0;
    D3D11_SUBRESOURCE_DATA vinitDataView;
    vinitDataView.pSysMem = viewVertices;

    hr = pd3dDevice->CreateBuffer(&vbdView, &vinitDataView, &viewVertexBuf);
    if (FAILED(hr)){
        char tmpstr[128];
        wsprintfA(tmpstr, "Error code : %lX", hr);
        OutputDebugStringA(tmpstr);
        //ErrorMessage(hr);
        return;
    }

    UINT viewIndices[] = {
        // the plane
        8, 9, 11,
        9, 10, 11,

        // front face
        0, 1, 2,
        0, 2, 3,

        // back face
        4, 6, 5,
        4, 7, 6,

        // left face
        4, 5, 1,
        4, 1, 0,

        // right face
        3, 2, 6,
        3, 6, 7,

        // top face
        1, 5, 6,
        1, 6, 2,

        // bottom face
        4, 0, 3,
        4, 3, 7
    };

    D3D11_BUFFER_DESC ibdView;
    ibdView.Usage = D3D11_USAGE_IMMUTABLE;
    ibdView.ByteWidth = sizeof(UINT) * 42;
    ibdView.BindFlags = D3D11_BIND_INDEX_BUFFER;
    ibdView.CPUAccessFlags = 0;
    ibdView.MiscFlags = 0;
    ibdView.StructureByteStride = 0;
    D3D11_SUBRESOURCE_DATA iinitDataView;
    iinitDataView.pSysMem = viewIndices;
    hr = pd3dDevice->CreateBuffer(&ibdView, &iinitDataView, &viewIndexBuf);
    if (FAILED(hr)){
        char tmpstr[128];
        wsprintfA(tmpstr, "Error code : %lX", hr);
        OutputDebugStringA(tmpstr);
        //ErrorMessage(hr);
        return;
    }


    UINT Strides[1];
    UINT Offsets[1];
    UINT stride = sizeof(Vertex);
    UINT offset = 0;

    //创建target对应texture的描述
    D3D11_TEXTURE2D_DESC resTexDesc;
    ZeroMemory(&resTexDesc, sizeof(resTexDesc));
    IDXGISwapChain*    pSwapChain = DXUTGetDXGISwapChain();
    ID3D11Texture2D *backBuffer(NULL);
    pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
    backBuffer->GetDesc(&resTexDesc);
    resTexDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    resTexDesc.BindFlags = 0;
    //resTexDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    //resTexDesc.Usage = D3D11_USAGE_STAGING;
    resTexDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
    backBuffer->Release();
    //target对应的texture2D
    hr = pd3dDevice->CreateTexture2D(&resTexDesc, NULL, &viewTex);
    if (FAILED(hr)){
        char tmpstr[128];
        wsprintfA(tmpstr, "Error code : %lX", hr);
        OutputDebugStringA(tmpstr);
        //ErrorMessage(hr);
        return;
    }

    //创建一个新的target描述
    //new render target view
    D3D11_RENDER_TARGET_VIEW_DESC resTVDesc;
    ZeroMemory(&resTVDesc, sizeof(resTVDesc));
    //g_pRenderTargetView[0]->GetDesc(&resTV);
    ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
    pRTV->GetDesc(&resTVDesc);
    resTVDesc.Format = resTexDesc.Format;
    resTVDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    resTVDesc.Texture2D.MipSlice = 0;

    //创建 targetview
    hr = pd3dDevice->CreateRenderTargetView(viewTex, &resTVDesc, &viewTV);
    if (FAILED(hr)){
        char tmpstr[128];
        wsprintfA(tmpstr, "Error code : %lX", hr);
        OutputDebugStringA(tmpstr);
        //ErrorMessage(hr);
        return;
    }

    //clear
    ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
    pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);
    float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    pd3dImmediateContext->ClearRenderTargetView(viewTV, ClearColor);
    pd3dImmediateContext->OMSetRenderTargets(1, &viewTV, pDSV);

    D3DXMATRIX mWorldViewProjection;
    D3DXVECTOR3 vLightDir;
    D3DXMATRIX mWorld;
    D3DXMATRIX mView;
    D3DXMATRIX mProj;
    D3DXMATRIX mTmp;

    //更新常数buffer
    // Per frame cb update
    D3D11_MAPPED_SUBRESOURCE MappedResource;
    V(pd3dImmediateContext->Map(g_pcbPSPerFrame, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource));
    CB_PS_PER_FRAME* pPerFrame = (CB_PS_PER_FRAME*)MappedResource.pData;
    float fAmbient = 0.1f;
    pPerFrame->m_vLightDirAmbient = D3DXVECTOR4(vecEyeView.x, vecEyeView.y, vecEyeView.z, fAmbient);
    pd3dImmediateContext->Unmap(g_pcbPSPerFrame, 0);
    pd3dImmediateContext->PSSetConstantBuffers(g_iCBPSPerFrameBind, 1, &g_pcbPSPerFrame);
    pd3dImmediateContext->IASetInputLayout(photonVertexLayout);
    pd3dImmediateContext->IASetVertexBuffers(0, 1, &viewVertexBuf, &stride, &offset);
    pd3dImmediateContext->IASetIndexBuffer(viewIndexBuf, DXGI_FORMAT_R32_UINT, 0);
    pd3dImmediateContext->VSSetShader(photonVertexShader, NULL, 0);
    pd3dImmediateContext->PSSetShader(photonPixelShaderPos, NULL, 0);

    // Set the per object constant data
    D3DXMATRIX m_orthoMatrix;
    XMMatrixScaling(0.5f, 0.5f, 0.5f);
    D3DXMatrixOrthoLH(&m_orthoMatrix, (float)windowWidth, (float)windowHeight, /*50.0f*/nearPlane, farPlane/*4000.0f*/);
    D3DXMATRIX scale;
    //D3DXMatrixScaling(&scale, 0.5f, 0.5f, 0.5f);
    D3DXMatrixScaling(&scale, meshScale, meshScale, meshScale);

    D3DXMATRIX dView;
    D3DXVECTOR3 dEye(vecEyeView.x, vecEyeView.y, vecEyeView.z);
    D3DXVECTOR3 dAt(0.0f, 0.0f, 0.0f);
    D3DXVECTOR3 dUp(0.0f, 0.0f, 1.0f);
    D3DXMatrixLookAtLH(&dView, &dEye, &dAt, &dUp);
    D3DXMATRIX dWorld;
    D3DXMatrixIdentity(&dWorld);
    D3DXMATRIX dProj;
    D3DXMatrixPerspectiveFovLH(&dProj, D3DX_PI / 4, 1.0f, 1.0f, 6000.0f);

    mWorldViewProjection = dWorld * dView * dProj;

    // VS Per object
    V(pd3dImmediateContext->Map(g_pcbVSPerObject, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource));
    CB_VS_PER_OBJECT* pVSPerObject = (CB_VS_PER_OBJECT*)MappedResource.pData;
    D3DXMatrixTranspose(&pVSPerObject->m_WorldViewProj, &mWorldViewProjection);
    D3DXMatrixTranspose(&pVSPerObject->m_World, &mWorld);
    pd3dImmediateContext->Unmap(g_pcbVSPerObject, 0);
    pd3dImmediateContext->VSSetConstantBuffers(g_iCBVSPerObjectBind, 1, &g_pcbVSPerObject);

    // PS Per object
    V(pd3dImmediateContext->Map(g_pcbPSPerObject, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource));
    CB_PS_PER_OBJECT* pPSPerObject = (CB_PS_PER_OBJECT*)MappedResource.pData;
    pPSPerObject->m_vObjectColor = D3DXVECTOR4(1, 1, 1, 1);
    pd3dImmediateContext->Unmap(g_pcbPSPerObject, 0);
    pd3dImmediateContext->PSSetConstantBuffers(g_iCBPSPerObjectBind, 1, &g_pcbPSPerObject);

    //Render
    D3D11_PRIMITIVE_TOPOLOGY PrimType;
    PrimType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    pd3dImmediateContext->IASetPrimitiveTopology(PrimType);
    pd3dImmediateContext->DrawIndexed(42, 0, 0);

    //将target复制出来
    D3D11_TEXTURE2D_DESC copyDesc;
    viewTex->GetDesc(&copyDesc);
    copyDesc.BindFlags = 0;
    copyDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    copyDesc.Usage = D3D11_USAGE_STAGING;
    copyDesc.BindFlags = 0;
    copyDesc.MiscFlags = 0;
    hr = pd3dDevice->CreateTexture2D(&copyDesc, NULL, &viewTexCopy);
    if (FAILED(hr))
        return;

    pd3dImmediateContext->CopyResource(viewTexCopy, viewTex);
    D3DX11SaveTextureToFile(pd3dImmediateContext, viewTexCopy, D3DX11_IFF_BMP, L"newview.bmp");
    D3D11_MAPPED_SUBRESOURCE frameResources;
    ZeroMemory(&frameResources, sizeof(D3D11_MAPPED_SUBRESOURCE));
    pd3dImmediateContext->Map(viewTexCopy, 0, D3D11_MAP_READ, 0, &frameResources);
    XMFLOAT4 * ptr = (XMFLOAT4*)(frameResources.pData);
    int count = 0;

    for (size_t i = 0; i < 512; i++)
    {
        for (size_t j = 0; j < 512; j++)
        {
            if (ptr[i * 512 + j].w > 0.5){
                posView[count][0] = ptr[i * 512 + j].x;
                posView[count][1] = ptr[i * 512 + j].y;
                posView[count][2] = ptr[i * 512 + j].z;
                count++;
            }
        }
    }

    pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);

    pd3dImmediateContext->ClearRenderTargetView(viewTV, ClearColor);
    pd3dImmediateContext->OMSetRenderTargets(1, &viewTV, pDSV);

    pd3dImmediateContext->VSSetShader(photonVertexShader, NULL, 0);
    pd3dImmediateContext->PSSetShader(photonPixelShaderDir, NULL, 0);

    D3DXMatrixOrthoLH(&m_orthoMatrix, (float)windowWidth, (float)windowHeight, /*50.0f*/nearPlane, farPlane/*4000.0f*/);
    D3DXMatrixScaling(&scale, meshScale, meshScale, meshScale);

    //Render
    //D3D11_PRIMITIVE_TOPOLOGY PrimType;
    PrimType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    pd3dImmediateContext->IASetPrimitiveTopology(PrimType);
    pd3dImmediateContext->DrawIndexed(42, 0, 0);

    //将target复制出来
    //D3D11_TEXTURE2D_DESC copyDesc;
    viewTex->GetDesc(&copyDesc);
    copyDesc.BindFlags = 0;
    copyDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    copyDesc.Usage = D3D11_USAGE_STAGING;
    copyDesc.BindFlags = 0;
    copyDesc.MiscFlags = 0;
    hr = pd3dDevice->CreateTexture2D(&copyDesc, NULL, &viewTexCopy);
    if (FAILED(hr))
        return;

    pd3dImmediateContext->CopyResource(viewTexCopy, viewTex);
    //D3D11_MAPPED_SUBRESOURCE frameResources;
    ZeroMemory(&frameResources, sizeof(D3D11_MAPPED_SUBRESOURCE));
    pd3dImmediateContext->Map(viewTexCopy, 0, D3D11_MAP_READ, 0, &frameResources);
    ptr = (XMFLOAT4*)(frameResources.pData);
    count = 0;

    for (size_t i = 0; i < 512; i++)
    {
        for (size_t j = 0; j < 512; j++)
        {
            if (ptr[i * 512 + j].w > 0.5){
                dirView[count][0] = ptr[i * 512 + j].x;
                dirView[count][1] = ptr[i * 512 + j].y;
                dirView[count][2] = ptr[i * 512 + j].z;
                float mul = dirView[count][0] * dirView[count][0] + dirView[count][1] * dirView[count][1] + dirView[count][2] * dirView[count][2];
                mul = sqrt(mul);
                dirView[count][0] = ptr[i * 512 + j].x / mul;
                dirView[count][1] = ptr[i * 512 + j].y / mul;
                dirView[count][2] = ptr[i * 512 + j].z / mul;
                count++;
            }

        }
    }
    pd3dImmediateContext->OMSetRenderTargets(1, &pRTV, pDSV);

}

unsigned char *ocTree;
unsigned char *nTree;
int buildTreeFlag = 1;
void buildTree(){
    if (!buildTreeFlag) return;
    int exp = 7;
    int len = 1 << exp;
    FILE *stream;
    if ((stream = fopen("voxel.data", "rb")) == NULL) /* open file TEST.$$$ */
    {
        fprintf(stderr, "Cannot open output file.\n");
        return;
    }
    nTree = (unsigned char *)malloc(len*len*len * 2 * sizeof(unsigned char));
    if (!nTree){
        fprintf(stderr, "CPU memory Malloc failed!");
        return;
    }
    fread(nTree, len * len *len * 2 * sizeof(unsigned char), 1, stream);
    fclose(stream);
    int a = constructOctree((unsigned char *)nTree, exp, &ocTree, 0.090);
    buildTreeFlag = 0;
}

//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------
size_t indexFrame = 0;
ID3D11Texture2D* background;
int renderCount = 0;

D3DXVECTOR3 p1;
D3DXVECTOR3 p2;
D3DXVECTOR3 p3;
D3DXVECTOR3 p4;

float planeLenBack = 900.0;
extern float TableZ;
void cal4Points(){
    float dx = vecAtView.x - vecEyeView.x;
    float dy = vecAtView.y - vecEyeView.y;
    float dz = vecAtView.z - vecEyeView.z;

    float muln = dx * dx + dy * dy + dz * dz;
    muln = sqrt(muln);

    dx /= muln;
    dy /= muln;
    dz /= muln;

    float centerX = vecEyeView.x + planeLenBack * dx;
    float centerY = vecEyeView.y + planeLenBack * dy;
    float centerZ = vecEyeView.z + planeLenBack * dz;

    float a;
    float b;
    if (abs(dy - 0.0) < 0.00000001){
        a = 0;
        b = 1.0;
    }
    else
    {
        a = 1.0f;
        b = -dx / dy;
    }

    muln = a * a + b * b;
    muln = sqrt(muln);

    a /= muln;
    b /= muln;

    float u = -dz * b;
    float v = dz * a;
    float w = dx * b - dy * a;

    muln = u*u + v*v + w*w;
    muln = sqrt(muln);

    u /= muln;
    v /= muln;
    w /= muln;

    p1.x = centerX - a * planeLenBack * 0.4142 - u * planeLenBack * 0.4142;
    p1.y = centerY - b * planeLenBack * 0.4142 - v * planeLenBack * 0.4142;
    p1.z = centerZ - w * planeLenBack * 0.4142;

    p2.x = centerX + a * planeLenBack * 0.4142 - u * planeLenBack * 0.4142;
    p2.y = centerY + b * planeLenBack * 0.4142 - v * planeLenBack * 0.4142;
    p2.z = centerZ - w * planeLenBack * 0.4142;

    p3.x = centerX + a * planeLenBack * 0.4142 + u * planeLenBack * 0.4142;
    p3.y = centerY + b * planeLenBack * 0.4142 + v * planeLenBack * 0.4142;
    p3.z = centerZ + w * planeLenBack * 0.4142;

    p4.x = centerX - a * planeLenBack * 0.4142 + u * planeLenBack * 0.4142;
    p4.y = centerY - b * planeLenBack * 0.4142 + v * planeLenBack * 0.4142;
    p4.z = centerZ + w * planeLenBack * 0.4142;
}

void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                  float fElapsedTime, void* pUserContext )
{
    planeLen = 780;
    planeLenBack = 800;

    if (!bRenderLight){
        vecEyeLight.x = -0.0f;
        vecEyeLight.y = -100.0f;
        vecEyeLight.z = 600.0f;
        renderLight(pd3dDevice, pd3dImmediateContext);
        bRenderLight = true;
    }
    if (!bRenderView){
        //vecAtView.x = 0.0f;
        //vecAtView.y = 0.0f;
        //vecAtView.z = 0.0f;
        //vecEyeView.x = -100.0f;
        //vecEyeView.y = 400.0f;
        //vecEyeView.z = -100.0f;
        renderView(pd3dDevice, pd3dImmediateContext);
        renderToBackTexture(pd3dDevice, pd3dImmediateContext);
        bRenderView = true;
    }
    if (!toBackTextureDown)
    {

        //renderToBackTexture(pd3dDevice, pd3dImmediateContext);
        toBackTextureDown = true;
    }

    //-------------------------------------
    IDXGISwapChain* pSwapChain = DXUTGetDXGISwapChain();
    ID3D11Texture2D *backBuffer(NULL);
    pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backBuffer));
    D3D11_TEXTURE2D_DESC texDesc;
    backBuffer->GetDesc(&texDesc);
    texDesc.BindFlags = 0;
    texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    texDesc.Usage = D3D11_USAGE_STAGING;
    texDesc.BindFlags = 0;
    texDesc.MiscFlags = 0;
    backBuffer->Release();

    texDesc.Width = 512;
    texDesc.Height = 512;
    HRESULT hr = pd3dDevice->CreateTexture2D(&texDesc, NULL, &tmp);
    D3D11_MAPPED_SUBRESOURCE frameResources;
    ZeroMemory(&frameResources, sizeof(D3D11_MAPPED_SUBRESOURCE));
    pd3dImmediateContext->Map(tmp, 0, D3D11_MAP_WRITE, 0, &frameResources);
    color *Frame = (color*)(frameResources.pData);

    TableZ = -200;
    //------------------------------------    
    int exp = 7;
    int len = 1 << exp;
    buildTree();
    //----------------
    int numPhoton = photonNumLight-10;
    float *table_d = 0;
    float *photondir=0, *photonrad=0, *photonpos=0;
    int *offset, *tableOffset, *flag;
    int TableR = 160;
    if (!marchBool)
    {
        direction = (float *)malloc(3 * len*len*len*sizeof(float));
        radiance = (float *)malloc(3 * len*len*len*sizeof(float));
        memset(direction, 0, 3 * len*len*len*sizeof(float));
        memset(radiance, 0, 3 * len*len*len*sizeof(float));
        photondir = (float *)&dirLight;
        photonpos = (float *)&posLight;
        photonrad = (float *)malloc(numPhoton * 3 * sizeof(float));
        for (int i = 0; i < numPhoton; i++)
        {
            photonrad[i * 3] = 0.001;
            photonrad[i * 3 + 1] = 0.001;
            photonrad[i * 3 + 2] = 0.001;
        }        
        marchPhoton(ocTree, nTree, direction, radiance, photondir, photonrad, photonpos, 7, numPhoton, 3.125, &table_d, 1024, TableR*2*1.0/1000, N);
        if (table) free(table);
        gaussian2D(table_d, &table, 1024, 13, 13.0 / 6, 5);
        marchBool = true;
    }    
    float *seephotondir = 0, *seephotonrad = 0, *seephotonpos = 0;
    seephotondir = (float *)&dirView;
    seephotonpos = (float *)&posView;
    seephotonrad = (float *)malloc(512 * 512 * 3 * sizeof(float));
    memset(seephotonrad, 0, 512 * 512 * 3 * sizeof(float));
    cal4Points();
    collectPhoton(ocTree, nTree, direction, radiance, seephotondir, seephotonrad, seephotonpos, 7, 512 * 512, 3.125,
        &offset, &tableOffset, &flag, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z, p4.x, p4.y, p4.z, N);
    float *prephotonrad_d;
    mapSeq(seephotonrad, &prephotonrad_d, sizeof(float), 512 * 512 * 3 );
    if (seephotonrad) free(seephotonrad);
    gaussian2D(prephotonrad_d, &seephotonrad, 512, 13, 13.0 / 6, 1);
    //-------------------------
    D3D11_MAPPED_SUBRESOURCE backResources;
    ZeroMemory(&backResources, sizeof(D3D11_MAPPED_SUBRESOURCE));
    pd3dImmediateContext->Map(resTexBackCopy, 0, D3D11_MAP_READ, 0, &backResources);
    color *ptcBack = (color*)(backResources.pData);
    D3DX11SaveTextureToFile(pd3dImmediateContext, resTexBackCopy, D3DX11_IFF_BMP, L"seeBack.bmp");

    int tmpR = 0;
    int tmpG = 0;
    int tmpB = 0;
    float *rad = seephotonrad;
    for (int j = 0; j < 512; j++){
        for (int k = 0; k < 512; k++){
            float scale =radScale;
            int tmpR = 0;
            int tmpG = 0;
            int tmpB = 0;
            if (flag[j * 512 + k] == 5)
                tmpR = tmpR;
            if (flag[j * 512 + k] == 1)
            {
                tmpR = (unsigned char)(rad[j * 512 * 3 + k * 3 + 0] * scale * 255) + (unsigned char)(ptcBack[offset[j * 512 + k]].r);
                tmpG = (unsigned char)(rad[j * 512 * 3 + k * 3 + 1] * scale * 255) + (unsigned char)(ptcBack[offset[j * 512 + k]].g);
                tmpB = (unsigned char)(rad[j * 512 * 3 + k * 3 + 2] * scale * 255) + (unsigned char)(ptcBack[offset[j * 512 + k]].b);
            }
            else if (flag[j * 512 + k] == 2)
            {                
                int tmp = tableOffset[j * 512 + k];
                int tx = tmp % (TableR*2);
                int ty = tmp / (TableR * 2);
                if (((tx / ((TableR * 2 / 8))) % 2) == ((ty / ((TableR * 2 / 8)) % 2)))
                {
                    tmpR = 165;
                    tmpG = 73;
                    tmpB = 40;
                }
                else
                {
                    tmpR = 176;
                    tmpG = 147;
                    tmpB = 114;
                }                
                int tablex = int(tx * (1024.0 / (TableR * 2)));
                int tabley = int(ty * (1024.0 / (TableR * 2)));
                float tableSC = 60;
                tmpR = tmpR + (unsigned int)(table[(tablex + tabley * 1024) * 3] * tableSC * 255);
                tmpG = tmpG + (unsigned int)(table[(tablex + tabley * 1024) * 3 + 1] * tableSC * 255);
                tmpB = tmpB + (unsigned int)(table[(tablex + tabley * 1024) * 3 + 2] * tableSC * 255);
                float sum = table[(tablex + tabley * 1024) * 3] + table[(tablex + tabley * 1024) * 3 + 1] + table[(tablex + tabley * 1024) * 3 + 2];
                if (bRenderShadow)
                    if (sum< 0.0001)
                    {
                        tmpR *= (0.1 + sum / 0.001 * 0.9);
                        tmpG *= (0.1 + sum / 0.001 * 0.9);
                        tmpB *= (0.1 + sum / 0.001 * 0.9);
                    }
                if (tmpR > 255)    tmpR = 255;
                if (tmpG > 255)    tmpG = 255;
                if (tmpB > 255)    tmpB = 255;
                tmpR += +(unsigned int)(rad[j * 512 * 3 + k * 3 + 0] * scale *5* 255);
                tmpG += +(unsigned int)(rad[j * 512 * 3 + k * 3 + 1] * scale *5* 255);
                tmpB += +(unsigned int)(rad[j * 512 * 3 + k * 3 + 2] * scale *5* 255);

            }
            else
            {
                tmpR = (unsigned char)(rad[j * 512 * 3 + k * 3 + 0] * scale * 255);
                tmpG = (unsigned char)(rad[j * 512 * 3 + k * 3 + 1] * scale * 255);
                tmpB = (unsigned char)(rad[j * 512 * 3 + k * 3 + 2] * scale * 255);
            }
            scale = 20;
            if (pureRad)
            {
                tmpR = (rad[j * 512 * 3 + k * 3 + 0] * scale * 255);
                tmpG =(rad[j * 512 * 3 + k * 3 + 1] * scale * 255);
                tmpB = (rad[j * 512 * 3 + k * 3 + 2] * scale * 255);
            }

            if (tmpR > 255)
                tmpR = 255;
            if (tmpG > 255)
                tmpG = 255;
            if (tmpB > 255)
                tmpB = 255;
            Frame[j * 512  +k].r= (unsigned char)tmpR;
            Frame[j * 512 + k].g = (unsigned char)tmpG;
            Frame[j * 512 + k].b = (unsigned char)tmpB;
        /*    if (false)if (offset[j * 512 + k] < 0){
                ptc[j * 512 * 4 + k * 4 + 0] = 255;
                ptc[j * 512 * 4 + k * 4 + 1] = 0;
                ptc[j * 512 * 4 + k * 4 + 2] = 0;
            }*/
        }
    }

    pd3dImmediateContext->Unmap(resTexBackCopy, 0);
    D3DX11SaveTextureToFile(pd3dImmediateContext, tmp, D3DX11_IFF_BMP, L"save.bmp");
    if (offset) free(offset);
    if (tableOffset) free(tableOffset);
    if (flag) free(flag);
//    if (direction) free(direction);
//    if (radiance) free(radiance);
    if (photonrad) free(photonrad);
    if (seephotonrad) free(seephotonrad);


    // If the settings dialog is being shown, then render it instead of rendering the app's scene
    if (g_D3DSettingsDlg.IsActive())
    {
        g_D3DSettingsDlg.OnRender(fElapsedTime);
        return;
    }

    // Clear the render target and depth stencil
    float ClearColor[4] = { 0.0f, 1.0f, 0.25f, 0.55f };

    ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
    pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);
    ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();

    pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0, 127);
    //    outputstencil(pDSV, pd3dDevice, pd3dImmediateContext);

    D3DXMATRIX mWorldViewProjection;
    D3DXVECTOR3 vLightDir;
    // Get the light direction
    vLightDir = g_LightControl.GetLightDirection();

    // Per frame cb update
    D3D11_MAPPED_SUBRESOURCE MappedResource;
    V(pd3dImmediateContext->Map(g_pcbPSPerFrame, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource));
    CB_PS_PER_FRAME* pPerFrame = (CB_PS_PER_FRAME*)MappedResource.pData;
    float fAmbient = 0.1f;
    pPerFrame->m_vLightDirAmbient = D3DXVECTOR4( vLightDir.x, vLightDir.y, vLightDir.z, fAmbient );
    pd3dImmediateContext->Unmap(g_pcbPSPerFrame, 0);
    pd3dImmediateContext->PSSetConstantBuffers(g_iCBPSPerFrameBind, 1, &g_pcbPSPerFrame);

    //Get the mesh
    //IA setup
    pd3dImmediateContext->IASetInputLayout(g_pVertexLayout11);
    UINT Strides[1];
    UINT Offsets[1];

    UINT stride = sizeof(SimpleVertex);
    UINT offset_ = 0;

    //====== ======
    pd3dImmediateContext->IASetVertexBuffers(0, 1, &imgVertexBuf, &stride, &offset_);
    pd3dImmediateContext->IASetIndexBuffer(imgIndexBuf, DXGI_FORMAT_R32_UINT, 0);

    // Set the shaders
    pd3dImmediateContext->VSSetShader(g_pVertexShader, NULL, 0);
    pd3dImmediateContext->PSSetShader(g_pPixelShader, NULL, 0);

    D3DXMATRIX m_orthoMatrix;                //正交投影矩阵  
    D3DXMatrixOrthoLH(&m_orthoMatrix, (float)windowWidth, (float)windowHeight, /*50.0f*/nearPlane, farPlane/*4000.0f*/);

    vecEyeImg.x = 0.0f;
    vecEyeImg.y = 600.0f;
    vecEyeImg.z = 0.0f;

    D3DXMATRIX dView;
    D3DXVECTOR3 dEye(vecEyeImg.x, vecEyeImg.y, vecEyeImg.z);
    D3DXVECTOR3 dAt(0.0f, 0.0f, 0.0f);
    D3DXVECTOR3 dUp(0.0f, 0.0f, 1.0f);
    D3DXMatrixLookAtLH(&dView, &dEye, &dAt, &dUp);
    D3DXMATRIX dWorld;
    D3DXMatrixIdentity(&dWorld);
    D3DXMatrixOrthoLH(&m_orthoMatrix, (float)windowWidth, (float)windowHeight, /*50.0f*/nearPlane, farPlane/*4000.0f*/);
    mWorldViewProjection = dWorld * dView * m_orthoMatrix;

    // VS Per object
    V(pd3dImmediateContext->Map(g_pcbVSPerObject, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource));
    CB_VS_PER_OBJECT* pVSPerObject = (CB_VS_PER_OBJECT*)MappedResource.pData;
    D3DXMatrixTranspose(&pVSPerObject->m_WorldViewProj, &mWorldViewProjection);
    D3DXMatrixTranspose(&pVSPerObject->m_World, &dWorld);
    pd3dImmediateContext->Unmap(g_pcbVSPerObject, 0);
    pd3dImmediateContext->VSSetConstantBuffers(g_iCBVSPerObjectBind, 1, &g_pcbVSPerObject);

    // PS Per object
    V(pd3dImmediateContext->Map(g_pcbPSPerObject, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource));
    CB_PS_PER_OBJECT* pPSPerObject = (CB_PS_PER_OBJECT*)MappedResource.pData;
    pPSPerObject->m_vObjectColor = D3DXVECTOR4(1, 1, 1, 1);
    pd3dImmediateContext->Unmap(g_pcbPSPerObject, 0);
    pd3dImmediateContext->PSSetConstantBuffers(g_iCBPSPerObjectBind, 1, &g_pcbPSPerObject);

    //Render
    SDKMESH_SUBSET* pSubset = NULL;
    D3D11_PRIMITIVE_TOPOLOGY PrimType;

    renderToBackTexture(pd3dDevice, pd3dImmediateContext);

    ID3D11Texture2D *newImg;
    D3D11_TEXTURE2D_DESC copyDesc;
    resTexBack->GetDesc(&copyDesc);
    //copyDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
    //copyDesc.Usage = D3D11_USAGE_STAGING;
    copyDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    copyDesc.MiscFlags = 0;
    hr = pd3dDevice->CreateTexture2D(&copyDesc, NULL, &newImg);
    if (FAILED(hr))
        return;
    pd3dImmediateContext->CopyResource(newImg, tmp);

    D3D11_SHADER_RESOURCE_VIEW_DESC backTexRVD;
    ZeroMemory(&backTexRVD, sizeof(backTexRVD));
    backTexRVD.Format = copyDesc.Format;
    backTexRVD.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    backTexRVD.Texture2D.MostDetailedMip = 0;
    backTexRVD.Texture2D.MipLevels = copyDesc.MipLevels;

    hr = pd3dDevice->CreateShaderResourceView(newImg, &backTexRVD, &imgTextureRVBack);
    if (FAILED(hr)){
        return;
    }
    pd3dImmediateContext->PSSetSamplers(0, 1, &g_pSamLinear);
    pd3dImmediateContext->PSSetShaderResources(0, 1, &g_pTextureRV);
    pd3dImmediateContext->PSSetShaderResources(1, 1, &imgTextureRVBack);

    PrimType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    pd3dImmediateContext->IASetPrimitiveTopology(PrimType);
    pd3dImmediateContext->DrawIndexed(6, 0, 0);

    DXUT_BeginPerfEvent(DXUT_PERFEVENTCOLOR, L"HUD / Stats");
    DXUT_EndPerfEvent();

}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
    g_DialogResourceManager.OnD3D11ReleasingSwapChain();
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{
    g_DialogResourceManager.OnD3D11DestroyDevice();
    g_D3DSettingsDlg.OnD3D11DestroyDevice();
    //CDXUTDirectionWidget::StaticOnD3D11DestroyDevice();
    DXUTGetGlobalResourceCache().OnDestroyDevice();
    SAFE_DELETE( g_pTxtHelper );

    g_Mesh11.Destroy();

    SAFE_RELEASE(g_pVertexLayout11);
    SAFE_RELEASE(photonVertexLayout);
    
    SAFE_RELEASE(g_pVertexBuffer);
    SAFE_RELEASE(g_pIndexBuffer);
    SAFE_RELEASE(g_pVertexShader);
    SAFE_RELEASE(g_pPixelShader);
    SAFE_RELEASE(g_pPixelShader1);
    SAFE_RELEASE(g_pSamLinear);
    SAFE_RELEASE(photonVertexShader);
    SAFE_RELEASE(photonPixelShaderPos);
    SAFE_RELEASE(photonPixelShaderDir);
    SAFE_RELEASE(g_pcbVSPerObject);
    SAFE_RELEASE(g_pcbPSPerObject);
    SAFE_RELEASE(g_pcbPSPerFrame);
    g_pTextureRV->Release();
}
