// 라이브러리
import axios from "axios";
import React, { useState, useEffect } from "react";
import { RouteProp } from "@react-navigation/native";
import { View, Text, Image } from "react-native";
import { useSelector } from "react-redux";
import LinearGradient from "react-native-linear-gradient";

// 타입
import { RootStackParamList } from "../../../App";
import { RootState } from "../../../store/store";

// 컴포넌트
import GoBackHeader from "../../../components/Header/GoBackHeader";
import MemoList from "../../../components/Modal/Memo/MemoList";
import MemoDetail from "../../../components/Modal/Memo/MemoDetail";
import LoadingOverlay from "../../../components/LoadingOverlay";

// 스타일
import { styles } from "./MemoStyle";

// 통신
import { BACKEND_URL } from "../../../util/http";

// 메뉴메모스크린 타입
type MenuMemoScreenProps = {
  route: RouteProp<RootStackParamList, "MenuMemo">;
};

// 메모 정보 타입
type MemoInfoType = {
  memoSeq: number;
  content: string;
  accessType: string;
  nickname: string;
  updatedAt: string;
  file: string | null;
  itemImage: string | null;
  isBookmarked: boolean;
};

const MemuMemo: React.FC<MenuMemoScreenProps> = ({ route }) => {
  // 메뉴 상태값 : saved-북마크 / all-전체 / my-내가 작성한
  const { menuStatus } = route.params;

  // 유저 아이디 (axios 통신용)
  const userId = useSelector((state: RootState) => state.userInfo.id);

  // 백엔드에서 주어준 정보
  const [memoData, setMemoData] = useState<MemoInfoType[]>([]);

  // AXIOS
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      let response;
      try {
        response = await axios.get(
          BACKEND_URL +
            // 쫀디기
            `/user/23/${
              menuStatus === "saved"
                ? "bookmarks"
                : menuStatus === "all"
                ? "memos"
                : "my-memos"
            }`
        );
        setMemoData(response.data);
      } catch (error) {
        console.log(error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // 메모리스트 처리를 위한 더미 함수
  const dummyHandler = () => {
    console.log("히히 더미 함수지롱~~");
  };

  // 로딩
  const [loading, setLoading] = useState(false);

  // 메모리스트 컴포넌트 상태관리
  const [isMemoList, setIsMemoList] = useState(true);

  // 메모디테일 컴포넌트 상태관리
  const [isMemoDetail, setIsMemoDetail] = useState(false);

  // 메모수정 컴포넌트 상태관리
  const [isMemoCreate, setIsMemoCreate] = useState(false);

  // 메모아이디 상태관리
  const [memoId, setMemoId] = useState(0);

  // 메모리스트 컴포넌트 닫고, 메모디테일 컴포넌트 열기
  const onDetailMemoHandler = (memoSeq: number) => {
    setIsMemoList(false);
    setIsMemoDetail(true);
    setMemoId(memoSeq);
  };

  // 메모디테일 -> 메모 수정
  const updateMemo = () => {
    setIsMemoDetail(false);
    setIsMemoCreate(true);
    console.log("메모 수정");
  };

  // 메모디테일 -> 메모 삭제
  const deleteMemo = () => {
    console.log("메모 삭제");
  };

  return (
    <>
      {loading ? (
        <LoadingOverlay />
      ) : (
        <>
          {memoData.length > 0 ? (
            // 메모 있을 때..
            <LinearGradient
              colors={["#F5F5F5", "#E9E9E9"]}
              start={{ x: 0, y: 0 }}
              end={{ x: 0, y: 1 }}
              style={{ flex: 1 }}
            >
              {isMemoList && (
                <>
                  <GoBackHeader />
                  <View>
                    {menuStatus === "saved" && (
                      <Text style={styles.title}>저장된 메모</Text>
                    )}
                    {menuStatus === "all" && (
                      <Text style={styles.title}>전체 메모</Text>
                    )}
                    {menuStatus === "my" && (
                      <Text style={styles.title}>내 메모</Text>
                    )}
                  </View>
                  <MemoList
                    onMemoWritePress={dummyHandler}
                    onMemoDetailPress={onDetailMemoHandler}
                    id={null}
                    memoStatus={menuStatus}
                  />
                </>
              )}
              {isMemoDetail && (
                <>
                  <GoBackHeader />
                  <MemoDetail
                    memoSeq={memoId}
                    onMemoUpdatePress={updateMemo}
                    onMemoDeletePress={deleteMemo}
                  />
                </>
              )}
              {isMemoCreate && (
                <View>
                  <Text>메모 생성을 해보장호!</Text>
                </View>
              )}
            </LinearGradient>
          ) : (
            // 메모 없을 때..
            <LinearGradient
              colors={["#F5F5F5", "#E9E9E9"]}
              start={{ x: 0, y: 0 }}
              end={{ x: 0, y: 1 }}
              style={{ flex: 1 }}
            >
              <GoBackHeader />
              {menuStatus === "saved" && (
                <View>
                  <Text style={styles.title}>저장된 메모</Text>
                  <View style={styles.container}>
                    <Image
                      style={styles.icon}
                      source={require("../../../assets/icons/memo_empty.png")}
                    />
                    <Text style={styles.empty}>저장된 메모가 없습니다</Text>
                  </View>
                </View>
              )}
              {menuStatus === "all" && (
                <View>
                  <Text style={styles.title}>전체 메모</Text>
                  <View style={styles.container}>
                    <Image
                      style={styles.icon}
                      source={require("../../../assets/icons/memo_empty.png")}
                    />
                    <Text style={styles.empty}>메모가 없습니다</Text>
                  </View>
                </View>
              )}
              {menuStatus === "my" && (
                <View>
                  <Text style={styles.title}>내 메모</Text>
                  <View style={styles.container}>
                    <Image
                      style={styles.icon}
                      source={require("../../../assets/icons/memo_empty.png")}
                    />
                    <Text style={styles.empty}>작성한 메모가 없습니다</Text>
                  </View>
                </View>
              )}
            </LinearGradient>
          )}
        </>
      )}
    </>
  );
};

export default MemuMemo;
