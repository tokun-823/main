import { useAppStore } from '../store/useAppStore';
import { CompositionBadge } from '../components/CompositionBadge';
import { CompositionType, Role } from '../types';

export const HeroCompositionTable = () => {
  const { heroes, compositions } = useAppStore();

  // ロール別にヒーローをグループ化
  const herosByRole = {
    tank: heroes.filter(h => h.role === 'tank'),
    damage: heroes.filter(h => h.role === 'damage'),
    support: heroes.filter(h => h.role === 'support'),
  };

  const roleLabels: Record<Role, string> = {
    tank: 'タンク',
    damage: 'ダメージ',
    support: 'サポート',
  };

  const compositionTypes: CompositionType[] = ['POKE', 'DIVE', 'RUSH'];

  // 構成タイプに対応しているかチェック
  const hasComposition = (heroCompositions: CompositionType[], type: CompositionType) => {
    return heroCompositions.includes(type);
  };

  // 統計情報を計算
  const stats = compositionTypes.map(comp => ({
    type: comp,
    count: heroes.filter(h => h.compositions.includes(comp)).length,
  }));

  return (
    <div className="min-h-screen bg-primary">
      <div className="container mx-auto px-4 py-8">
        {/* ヘッダー */}
        <div className="mb-8">
          <h1 className="text-4xl font-black text-white mb-2">ヒーロー構成タイプ一覧表</h1>
          <p className="text-text-sub">全50体のヒーローのPOKE / DIVE / RUSH対応状況を一覧で確認</p>
        </div>

        {/* 統計情報 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {stats.map(stat => {
            const compInfo = compositions[stat.type];
            return (
              <div
                key={stat.type}
                className="bg-surface border-2 border-border rounded-lg p-6"
                style={{
                  borderColor: compInfo.color,
                  backgroundColor: compInfo.color + '10',
                }}
              >
                <div className="flex items-center justify-between mb-2">
                  <CompositionBadge type={stat.type} size="lg" />
                  <span className="text-3xl font-black text-white">{stat.count}体</span>
                </div>
                <p className="text-sm text-text-sub">{compInfo.description}</p>
              </div>
            );
          })}
        </div>

        {/* 凡例 */}
        <div className="bg-surface border border-border rounded-lg p-6 mb-8">
          <h3 className="text-lg font-bold text-white mb-4">凡例</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded bg-accent flex items-center justify-center text-white font-bold">✓</div>
              <span className="text-text-sub">対応している構成タイプ</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded bg-primary border border-border flex items-center justify-center text-text-sub">−</div>
              <span className="text-text-sub">対応していない構成タイプ</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex gap-1">
                <div className="w-6 h-6 rounded" style={{ backgroundColor: compositions.POKE.color + '40' }}></div>
                <div className="w-6 h-6 rounded" style={{ backgroundColor: compositions.DIVE.color + '40' }}></div>
                <div className="w-6 h-6 rounded" style={{ backgroundColor: compositions.RUSH.color + '40' }}></div>
              </div>
              <span className="text-text-sub">複数の構成に対応</span>
            </div>
          </div>
        </div>

        {/* ロール別一覧表 */}
        <div className="space-y-8">
          {(['tank', 'damage', 'support'] as const).map(role => (
            <div key={role} className="bg-surface border border-border rounded-lg overflow-hidden">
              {/* ロールヘッダー */}
              <div className="bg-primary border-b border-border p-4">
                <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                  {roleLabels[role]}
                  <span className="text-sm font-normal text-text-sub">
                    ({herosByRole[role].length}体)
                  </span>
                </h2>
              </div>

              {/* テーブル */}
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-primary border-b border-border">
                      <th className="text-left p-4 text-white font-bold w-1/4">ヒーロー名</th>
                      {compositionTypes.map(comp => (
                        <th key={comp} className="text-center p-4 w-1/4">
                          <CompositionBadge type={comp} size="md" />
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {herosByRole[role].map((hero, index) => (
                      <tr
                        key={hero.id}
                        className={`border-b border-border transition-colors hover:bg-primary ${
                          index % 2 === 0 ? 'bg-surface' : 'bg-primary bg-opacity-30'
                        }`}
                      >
                        {/* ヒーロー名 */}
                        <td className="p-4">
                          <div className="flex items-center gap-3">
                            <div
                              className="w-12 h-12 bg-primary border-2 border-border rounded flex items-center justify-center overflow-hidden relative"
                              style={{
                                clipPath: 'polygon(10% 0%, 100% 0%, 90% 100%, 0% 100%)',
                              }}
                            >
                              <img 
                                src={`/heroes/${hero.id}.avif`}
                                alt={hero.name}
                                className="absolute inset-0 w-full h-full object-cover opacity-60"
                                onError={(e) => {
                                  (e.target as HTMLImageElement).style.display = 'none';
                                }}
                              />
                            </div>
                            <div>
                              <div className="text-white font-bold">{hero.name}</div>
                              <div className="text-xs text-text-sub">{hero.nameEn}</div>
                            </div>
                          </div>
                        </td>

                        {/* 構成タイプ対応状況 */}
                        {compositionTypes.map(comp => {
                          const isSupported = hasComposition(hero.compositions, comp);
                          const compInfo = compositions[comp];
                          
                          return (
                            <td key={comp} className="text-center p-4">
                              {isSupported ? (
                                <div className="flex justify-center">
                                  <div
                                    className="w-16 h-16 rounded-lg flex items-center justify-center font-black text-2xl text-white shadow-lg transition-transform hover:scale-110"
                                    style={{
                                      backgroundColor: compInfo.color,
                                      boxShadow: `0 4px 12px ${compInfo.color}60`,
                                    }}
                                  >
                                    ✓
                                  </div>
                                </div>
                              ) : (
                                <div className="flex justify-center">
                                  <div className="w-16 h-16 rounded-lg flex items-center justify-center bg-primary border border-border text-text-sub text-2xl">
                                    −
                                  </div>
                                </div>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* ロール別統計 */}
              <div className="bg-primary border-t border-border p-4">
                <div className="flex items-center justify-end gap-6 text-sm">
                  {compositionTypes.map(comp => {
                    const count = herosByRole[role].filter(h => h.compositions.includes(comp)).length;
                    return (
                      <div key={comp} className="flex items-center gap-2">
                        <CompositionBadge type={comp} size="sm" />
                        <span className="text-white font-bold">{count}体</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* 補足情報 */}
        <div className="mt-8 bg-surface border border-border rounded-lg p-6">
          <h3 className="text-xl font-bold text-white mb-4">💡 構成タイプについて</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <CompositionBadge type="POKE" size="md" />
                <span className="text-white font-bold">ポーク構成</span>
              </div>
              <p className="text-sm text-text-sub leading-relaxed">
                遠距離から敵を削る戦術。シールドタンクと長距離DPSを中心に、安全な距離から攻撃を行います。
              </p>
            </div>
            <div>
              <div className="flex items-center gap-2 mb-2">
                <CompositionBadge type="DIVE" size="md" />
                <span className="text-white font-bold">ダイブ構成</span>
              </div>
              <p className="text-sm text-text-sub leading-relaxed">
                高機動力で敵後衛に飛び込む戦術。全員で同時に飛び込み、素早くキルを取って離脱します。
              </p>
            </div>
            <div>
              <div className="flex items-center gap-2 mb-2">
                <CompositionBadge type="RUSH" size="md" />
                <span className="text-white font-bold">ラッシュ構成</span>
              </div>
              <p className="text-sm text-text-sub leading-relaxed">
                スピードブーストで接近し、近距離火力で圧倒する戦術。まとまって行動することが重要です。
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
