package com.tjjhtjh.memorise.domain.memo.repository;

import com.tjjhtjh.memorise.domain.memo.repository.entity.Bookmark;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MyMemoResponse;

import java.util.List;
import java.util.Optional;

public interface BookmarkRepositoryCustom {
   Optional<Bookmark> findByMemoAndUser(Long memoId, Long userSeq);
   List<Bookmark> fakeDeleteMemoBookmarkList(Long memoId);
   List<MyMemoResponse> isBookmarkTrueList(Long userSeq);
   Boolean bookmarkBoolean(Long memoId, Long userSeq);

}
