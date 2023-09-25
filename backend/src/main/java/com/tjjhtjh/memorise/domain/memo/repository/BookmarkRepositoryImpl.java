package com.tjjhtjh.memorise.domain.memo.repository;

import com.querydsl.core.types.ExpressionUtils;
import com.querydsl.core.types.Projections;
import com.querydsl.core.types.dsl.CaseBuilder;
import com.querydsl.jpa.JPAExpressions;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Bookmark;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MyMemoResponse;
import org.springframework.data.jpa.repository.support.QuerydslRepositorySupport;

import java.util.List;
import java.util.Optional;

import static com.tjjhtjh.memorise.domain.memo.repository.entity.QBookmark.bookmark;
import static com.tjjhtjh.memorise.domain.memo.repository.entity.QMemo.memo;
import static com.tjjhtjh.memorise.domain.tag.repository.entity.QTaggedUser.taggedUser;


public class BookmarkRepositoryImpl extends QuerydslRepositorySupport implements BookmarkRepositoryCustom {

    private final JPAQueryFactory queryFactory;

    public BookmarkRepositoryImpl(JPAQueryFactory jpaQueryFactory){
        super(Bookmark.class);
        this.queryFactory = jpaQueryFactory;
    }

    @Override
    public Optional<Bookmark> findByMemoAndUser(Long memoId, String email) {
        return queryFactory.selectFrom(bookmark)
                .where(bookmark.user.email.eq(email).and(bookmark.memo.memoSeq.eq(memoId)))
                .stream().findAny();
    }

    @Override
    public List<Bookmark> bookmarkExistCheck(Long memoId, Long userSeq) {
        return queryFactory.selectFrom(bookmark).where(bookmark.user.userSeq.eq(userSeq).and(bookmark.memo.memoSeq.eq(memoId)))
                .fetch();
    }

    @Override
    public List<MyMemoResponse> isBookmarkTrueList(Long userSeq) {
        return queryFactory.select(Projections.fields
                        (MyMemoResponse.class, memo.memoSeq, memo.user.nickname, memo.updatedAt, memo.content, memo.accessType, memo.file, memo.item.itemImage
                                , ExpressionUtils.as(new CaseBuilder().when(JPAExpressions.selectOne().from(bookmark)
                                                .where(bookmark.memo.memoSeq.eq(memo.memoSeq).and(bookmark.user.userSeq.eq(userSeq)))
                                                .exists()).then(true).otherwise(false), "isBookmarked")))
                .from(memo)
                .leftJoin(memo.user)
                .leftJoin(memo.item)
                .leftJoin(bookmark).on(memo.memoSeq.eq(bookmark.memo.memoSeq).and(bookmark.user.userSeq.eq(userSeq)))
                .where(memo.isDeleted.eq(0))
                .groupBy(memo.memoSeq)
                .having(new CaseBuilder().when(JPAExpressions.selectOne().from(bookmark)
                                .where(bookmark.memo.memoSeq.eq(memo.memoSeq).and(bookmark.user.userSeq.eq(userSeq)))
                                .exists()).then(true).otherwise(false).eq(true))
                .orderBy(memo.updatedAt.desc())
                .fetch();
    }
}
